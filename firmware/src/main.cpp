#include <Arduino.h>
#include "esp_camera.h"
#include "img_converters.h"
#include <esp_heap_caps.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_settings.h"
#include "custom_object_model_data.h"

// AI-Thinker ESP32-CAM Pin Definitions
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define LED_PIN           33  // Active-low red status LED
#define FLASH_PIN          4  // Active-high onboard flashlight LED

#define SERIAL_BAUD_RATE  460800

// Operating Modes
enum OperationMode {
  MODE_HYBRID = 0,       // Classical CV Saliency / Motion triggers Localized Neural Recognition
  MODE_CONTINUOUS_AI = 1,// Neural Network runs continuously on every frame (Full Center FOV)
  MODE_PURE_CV = 2       // High-speed Classical CV only (Peak FPS)
};

static OperationMode current_mode = MODE_HYBRID;
static int confidence_threshold_pct = 45; // 45% confidence threshold for object detection
static bool flash_state = false;
static bool send_frame_requested = false;
static bool continuous_video_stream = true;

// TFLM Globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;

  // Tensor arena size: 96 KB in PSRAM
  constexpr int kTensorArenaSize = 96 * 1024;
  static uint8_t* tensor_arena = nullptr;
  
  // Op resolver for DeskObjectNet INT8 model
  static tflite::MicroMutableOpResolver<14> micro_op_resolver;
}

// Classical CV State
static uint8_t prev_frame[96 * 96];
static bool has_prev_frame = false;

// Statistics & Metrics
static uint32_t total_frames = 0;
static uint32_t total_inferences = 0;
static uint32_t total_object_detections = 0;
static uint32_t last_fps_time = 0;
static uint32_t fps_frame_count = 0;
static float current_fps = 0.0f;
static uint32_t last_inference_duration_ms = 0;

// Camera Initialization
bool init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; // RGB 16-bit color capture
  config.frame_size = FRAMESIZE_96X96;    // 96x96 sensor resolution
  config.jpeg_quality = 12;
  config.fb_count = 2; // Double buffering
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERR] Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 1);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
  }
  return true;
}

// TFLM Initialization
bool init_tflm() {
  model = tflite::GetModel(g_custom_object_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERR] Model schema %ld != supported %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Allocate tensor arena in PSRAM
  if (psramFound()) {
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (!tensor_arena) {
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT);
  }
  if (!tensor_arena) {
    Serial.println("[ERR] Failed to allocate tensor arena!");
    return false;
  }

  // Register operators for DeskObjectNet INT8
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddPadV2();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddRelu6();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERR] TFLM AllocateTensors() failed!");
    return false;
  }

  model_input = interpreter->input(0);
  if (!model_input) {
    Serial.println("[ERR] Failed to get model input tensor!");
    return false;
  }

  Serial.printf("[TFLM] DeskObjectNet loaded! Input bytes: %d, Arena used: %d bytes\n",
                model_input->bytes, interpreter->arena_used_bytes());
  return true;
}

// Classical CV Color Saliency & Motion Analysis
struct SaliencyResult {
  bool has_candidate;
  int crop_x;             // Crop origin in 96x96 frame
  int crop_y;
  int crop_w;
  int crop_h;
  const char* proposal_type;
  float motion_score;
  int centroid_x;
  int centroid_y;
};

SaliencyResult analyze_saliency_and_motion(const uint8_t* rgb565_buf) {
  SaliencyResult res = {false, 0, 0, 48, 48, "none", 0.0f, 48, 48};
  
  uint32_t total_diff = 0;
  uint32_t weighted_x = 0;
  uint32_t weighted_y = 0;
  uint32_t diff_pixel_count = 0;
  
  uint32_t cyan_count = 0;
  uint32_t cyan_x = 0, cyan_y = 0;
  
  uint32_t orange_count = 0;
  uint32_t orange_x = 0, orange_y = 0;

  for (int y = 0; y < 96; y++) {
    int row_offset = y * 96 * 2;
    for (int x = 0; x < 96; x++) {
      int idx = row_offset + x * 2;
      uint16_t pix = ((uint16_t)rgb565_buf[idx] << 8) | rgb565_buf[idx + 1];
      
      // Fast RGB channel extraction
      uint8_t r = ((pix >> 11) & 0x1F) << 3;
      uint8_t g = ((pix >> 5) & 0x3F) << 2;
      uint8_t b = (pix & 0x1F) << 3;
      
      // Fast Luminance
      uint8_t lum = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
      
      // Motion differencing
      int p_idx = y * 96 + x;
      if (has_prev_frame) {
        int diff = abs((int)lum - (int)prev_frame[p_idx]);
        if (diff > 18) {
          total_diff += diff;
          weighted_x += x * diff;
          weighted_y += y * diff;
          diff_pixel_count++;
        }
      }
      prev_frame[p_idx] = lum;
      
      // Color Saliency Heuristics:
      // 1. Cyan/Mint: G > 100, B > 100, R < 90, |G - B| < 40
      if (g > 90 && b > 90 && r < 90) {
        cyan_count++;
        cyan_x += x;
        cyan_y += y;
      }
      // 2. Orange/Cream ring: R > 130, G > 70, G < 140, B < 60
      if (r > 130 && g > 65 && g < 140 && b < 60) {
        orange_count++;
        orange_x += x;
        orange_y += y;
      }
    }
  }
  has_prev_frame = true;

  if (diff_pixel_count > 0 && total_diff > 0) {
    res.motion_score = ((float)diff_pixel_count / (96.0f * 96.0f)) * 100.0f;
    res.centroid_x = (int)(weighted_x / total_diff);
    res.centroid_y = (int)(weighted_y / total_diff);
  }

  // Saliency decision
  if (cyan_count >= 15) {
    res.has_candidate = true;
    res.proposal_type = "cyan_earbuds";
    int cx = cyan_x / cyan_count;
    int cy = cyan_y / cyan_count;
    res.crop_x = constrain(cx - 24, 0, 48);
    res.crop_y = constrain(cy - 24, 0, 48);
  } else if (orange_count >= 12) {
    res.has_candidate = true;
    res.proposal_type = "orange_cream";
    int cx = orange_x / orange_count;
    int cy = orange_y / orange_count;
    res.crop_x = constrain(cx - 24, 0, 48);
    res.crop_y = constrain(cy - 24, 0, 48);
  } else if (res.motion_score > 3.0f) {
    res.has_candidate = true;
    res.proposal_type = "motion_cluster";
    res.crop_x = constrain(res.centroid_x - 24, 0, 48);
    res.crop_y = constrain(res.centroid_y - 24, 0, 48);
  }
  return res;
}

// Preprocessing: Extracts 48x48x3 RGB crop and quantizes directly to model INT8 tensor
void preprocess_crop_to_tensor(const uint8_t* rgb565_buf, int crop_x, int crop_y, int crop_w, int crop_h) {
  int8_t* tensor_ptr = model_input->data.int8;
  
  // Scale factor: crop_w x crop_h -> 48 x 48
  float scale_x = (float)crop_w / 48.0f;
  float scale_y = (float)crop_h / 48.0f;

  for (int y = 0; y < 48; y++) {
    int src_y = crop_y + (int)(y * scale_y);
    src_y = constrain(src_y, 0, 95);
    int row_offset = src_y * 96 * 2;
    
    for (int x = 0; x < 48; x++) {
      int src_x = crop_x + (int)(x * scale_x);
      src_x = constrain(src_x, 0, 95);
      
      int pix_idx = row_offset + src_x * 2;
      uint16_t pix = ((uint16_t)rgb565_buf[pix_idx] << 8) | rgb565_buf[pix_idx + 1];
      
      // Extract RGB channels
      uint8_t r_u8 = ((pix >> 11) & 0x1F) << 3;
      uint8_t g_u8 = ((pix >> 5) & 0x3F) << 2;
      uint8_t b_u8 = (pix & 0x1F) << 3;
      
      // Quantize to INT8 [-128, 127] (scale = 1/128, zero_point = 0)
      int out_idx = (y * 48 + x) * 3;
      tensor_ptr[out_idx + 0] = (int8_t)(r_u8 - 128);
      tensor_ptr[out_idx + 1] = (int8_t)(g_u8 - 128);
      tensor_ptr[out_idx + 2] = (int8_t)(b_u8 - 128);
    }
  }
}

// Inference Result Structure
struct ObjectInferenceResult {
  int top_class_id;
  const char* top_class_name;
  float top_confidence;
  float probabilities[kCategoryCount];
  uint32_t duration_ms;
  bool success;
};

ObjectInferenceResult run_desk_object_inference() {
  ObjectInferenceResult res;
  res.top_class_id = 0;
  res.top_class_name = kCategoryLabels[0];
  res.top_confidence = 0.0f;
  res.duration_ms = 0;
  res.success = false;

  if (!model_input || !interpreter) return res;

  uint32_t t_start = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t t_dur = millis() - t_start;
  res.duration_ms = t_dur;
  last_inference_duration_ms = t_dur;
  total_inferences++;

  if (invoke_status != kTfLiteOk) {
    Serial.println("[ERR] interpreter->Invoke() failed!");
    return res;
  }

  TfLiteTensor* output = interpreter->output(0);
  float max_score = -1000.0f;
  float dequant_logits[kCategoryCount];
  float exp_sum = 0.0f;

  for (int i = 0; i < kCategoryCount; i++) {
    int8_t q_val = output->data.int8[i];
    float logit = (q_val - output->params.zero_point) * output->params.scale;
    dequant_logits[i] = logit;
    if (logit > max_score) max_score = logit;
  }

  // Softmax
  for (int i = 0; i < kCategoryCount; i++) {
    float ex = expf(dequant_logits[i] - max_score);
    res.probabilities[i] = ex;
    exp_sum += ex;
  }
  for (int i = 0; i < kCategoryCount; i++) {
    res.probabilities[i] /= exp_sum;
  }

  // Find top class
  int best_id = 0;
  float best_conf = 0.0f;
  for (int i = 0; i < kCategoryCount; i++) {
    if (res.probabilities[i] > best_conf) {
      best_conf = res.probabilities[i];
      best_id = i;
    }
  }

  res.top_class_id = best_id;
  res.top_class_name = kCategoryLabels[best_id];
  res.top_confidence = best_conf;
  res.success = true;

  if (best_id > 0 && (best_conf * 100.0f) >= confidence_threshold_pct) {
    total_object_detections++;
  }
  return res;
}

// Command Parser
static char cmd_buffer[64];
static size_t cmd_idx = 0;

void handle_command(const char* cmd) {
  if (strcmp(cmd, "MODE=HYBRID") == 0) {
    current_mode = MODE_HYBRID;
    Serial.println("[CMD] OK Mode set to HYBRID (Classical Saliency + Localized Neural Inference)");
  } else if (strcmp(cmd, "MODE=AI") == 0) {
    current_mode = MODE_CONTINUOUS_AI;
    Serial.println("[CMD] OK Mode set to CONTINUOUS_AI (Full-frame on every cycle)");
  } else if (strcmp(cmd, "MODE=CV") == 0) {
    current_mode = MODE_PURE_CV;
    Serial.println("[CMD] OK Mode set to PURE_CV (High-speed color/motion tracking)");
  } else if (strncmp(cmd, "THRESH=", 7) == 0) {
    int th = atoi(cmd + 7);
    if (th >= 10 && th <= 100) {
      confidence_threshold_pct = th;
      Serial.printf("[CMD] OK Confidence threshold set to %d%%\n", confidence_threshold_pct);
    }
  } else if (strcmp(cmd, "STREAM=1") == 0) {
    continuous_video_stream = true;
    Serial.println("[CMD] OK Continuous Video Streaming ON");
  } else if (strcmp(cmd, "STREAM=0") == 0) {
    continuous_video_stream = false;
    Serial.println("[CMD] OK Continuous Video Streaming OFF");
  } else if (strcmp(cmd, "FLASH=1") == 0) {
    flash_state = true;
    digitalWrite(FLASH_PIN, HIGH);
    Serial.println("[CMD] Flashlight ON");
  } else if (strcmp(cmd, "FLASH=0") == 0) {
    flash_state = false;
    digitalWrite(FLASH_PIN, LOW);
    Serial.println("[CMD] Flashlight OFF");
  } else if (strcmp(cmd, "GET_FRAME") == 0) {
    send_frame_requested = true;
    Serial.println("[CMD] OK Single frame requested");
  } else if (strcmp(cmd, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(cmd, "STATUS") == 0) {
    Serial.printf("{\"status\":\"ok\",\"mode\":%d,\"model\":\"DeskObjectNet_INT8\",\"thresh\":%d,\"stream\":%d,\"fps\":%.1f,\"heap\":%lu,\"psram\":%lu,\"inferences\":%lu,\"detections\":%lu}\n",
                  (int)current_mode, confidence_threshold_pct, continuous_video_stream ? 1 : 0, current_fps,
                  esp_get_free_heap_size(), ESP.getFreePsram(), total_inferences, total_object_detections);
  } else if (strcmp(cmd, "RESET") == 0) {
    Serial.println("[CMD] Restarting device...");
    delay(100);
    ESP.restart();
  }
}

void process_serial_commands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmd_idx > 0) {
        cmd_buffer[cmd_idx] = '\0';
        handle_command(cmd_buffer);
        cmd_idx = 0;
      }
    } else if (cmd_idx < sizeof(cmd_buffer) - 1) {
      cmd_buffer[cmd_idx++] = c;
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  delay(400);

  pinMode(LED_PIN, OUTPUT);
  pinMode(FLASH_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);   // Status LED on
  digitalWrite(FLASH_PIN, LOW);  // Flash off

  Serial.println("\n==============================================");
  Serial.println(" ESP32-CAM Desk Object AI Vision System v2.0  ");
  Serial.println(" Model: DeskObjectNet INT8 (6 Classes)        ");
  Serial.println("==============================================");
  Serial.printf("Chip: ESP32-D0WD-V3 @ %d MHz | Reset Reason: %d\n", 
                (int)getCpuFrequencyMhz(), (int)esp_reset_reason());
  Serial.printf("Total Free Heap: %lu bytes | PSRAM: %s (%lu bytes free)\n",
                esp_get_free_heap_size(), psramFound() ? "YES" : "NO", ESP.getFreePsram());

  // Initialize Camera
  if (!init_camera()) {
    Serial.println("[FATAL] Camera initialization failed! Halting.");
    while (true) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(200);
    }
  }
  Serial.println("[CAMERA] OV3660/OV2640 Initialized (96x96 RGB565 Double-Buffered)");

  // Initialize TFLM
  if (!init_tflm()) {
    Serial.println("[FATAL] TFLM Model initialization failed! Halting.");
    while (true) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(500);
    }
  }

  digitalWrite(LED_PIN, HIGH); // Status LED off (ready)
  Serial.println("[READY] System initialized and streaming telemetry.\n");
  last_fps_time = millis();
}

void loop() {
  process_serial_commands();

  // Capture frame from camera
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("{\"error\":\"camera_fb_get_failed\"}");
    delay(20);
    return;
  }

  total_frames++;
  fps_frame_count++;

  // Step 1: Classical CV Saliency and Motion
  SaliencyResult saliency = analyze_saliency_and_motion(fb->buf);

  // Step 2: Determine if and where Neural Network runs
  bool run_ai = false;
  int crop_x = 24, crop_y = 24, crop_w = 48, crop_h = 48; // default center FOV

  if (current_mode == MODE_CONTINUOUS_AI) {
    run_ai = true;
    crop_x = 24; crop_y = 24; crop_w = 48; crop_h = 48;
  } else if (current_mode == MODE_HYBRID) {
    if (saliency.has_candidate) {
      run_ai = true;
      crop_x = saliency.crop_x;
      crop_y = saliency.crop_y;
      crop_w = 48;
      crop_h = 48;
    } else if ((total_frames % 20) == 0) {
      // Periodic background scan
      run_ai = true;
      crop_x = 24; crop_y = 24; crop_w = 48; crop_h = 48;
    }
  }

  ObjectInferenceResult ai_res;
  ai_res.top_class_id = 0;
  ai_res.top_class_name = kCategoryLabels[0];
  ai_res.top_confidence = 0.0f;
  ai_res.duration_ms = 0;
  ai_res.success = false;

  if (run_ai) {
    digitalWrite(LED_PIN, LOW); // LED indicates inference active
    preprocess_crop_to_tensor(fb->buf, crop_x, crop_y, crop_w, crop_h);
    ai_res = run_desk_object_inference();
    digitalWrite(LED_PIN, HIGH);
  }

  // Calculate rolling FPS every 1000ms
  uint32_t now = millis();
  if (now - last_fps_time >= 1000) {
    current_fps = (fps_frame_count * 1000.0f) / (now - last_fps_time);
    fps_frame_count = 0;
    last_fps_time = now;
  }

  // Output structured telemetry JSON
  int conf_pct = (int)(ai_res.top_confidence * 100.0f);
  bool object_detected = (run_ai && ai_res.top_class_id > 0 && conf_pct >= confidence_threshold_pct);

  // Send real-time camera video frame if streaming enabled or requested
  if (continuous_video_stream || send_frame_requested) {
    send_frame_requested = false;
    uint8_t *jpg_buf = nullptr;
    size_t jpg_len = 0;
    if (frame2jpg(fb, 75, &jpg_buf, &jpg_len)) {
      if (jpg_buf && jpg_len > 0) {
        const uint8_t MAGIC_HEADER[4] = {0xAA, 0x55, 0xAA, 0x55};
        const uint8_t MAGIC_FOOTER[2] = {0x55, 0xAA};
        uint32_t len = (uint32_t)jpg_len;
        Serial.write(MAGIC_HEADER, 4);
        Serial.write((const uint8_t*)&len, 4);
        Serial.write(jpg_buf, jpg_len);
        Serial.write(MAGIC_FOOTER, 2);
        free(jpg_buf);
      }
    }
  }

  Serial.printf("{\"f\":%lu,\"m\":%d,\"ai\":%d,\"cid\":%d,\"cls\":\"%s\",\"conf\":%d,\"det\":%d,\"bx\":%d,\"by\":%d,\"bw\":%d,\"bh\":%d,\"prop\":\"%s\",\"mot\":%.1f,\"lat\":%lu,\"fps\":%.1f,\"hp\":%lu,\"ps\":%lu}\n",
                total_frames, (int)current_mode, run_ai ? 1 : 0,
                ai_res.top_class_id, ai_res.top_class_name, conf_pct, object_detected ? 1 : 0,
                crop_x, crop_y, crop_w, crop_h, saliency.proposal_type,
                saliency.motion_score, ai_res.duration_ms, current_fps,
                esp_get_free_heap_size(), ESP.getFreePsram());

  // Return camera frame buffer
  esp_camera_fb_return(fb);

  // Yield to FreeRTOS watchdog & idle task
  vTaskDelay(1);
}
