import os
import shutil

src_tflite = r"C:\All_Projects\esp32_cam_ai_system\models\candidate_models\sensor_matched_tflite\sensor_matched_model_full_integer_quant.tflite"
dst_tflite = r"C:\All_Projects\esp32_cam_ai_system\models\deployment\custom_desk_object_model_int8.tflite"
shutil.copyfile(src_tflite, dst_tflite)

with open(dst_tflite, "rb") as f:
    bytes_data = f.read()

c_header_path = r"C:\All_Projects\esp32_cam_ai_system\models\deployment\custom_object_model_data.h"
c_source_path = r"C:\All_Projects\esp32_cam_ai_system\models\deployment\custom_object_model_data.cc"

fw_header_path = r"C:\All_Projects\esp32_cam_ai_system\firmware\include\custom_object_model_data.h"
fw_source_path = r"C:\All_Projects\esp32_cam_ai_system\firmware\src\custom_object_model_data.cc"

header_content = """#ifndef CUSTOM_OBJECT_MODEL_DATA_H_
#define CUSTOM_OBJECT_MODEL_DATA_H_

extern const unsigned char g_custom_object_model_data[];
extern const int g_custom_object_model_data_len;

#endif  // CUSTOM_OBJECT_MODEL_DATA_H_
"""

source_content = f"""#include "custom_object_model_data.h"

// Model: DeskObjectNet INT8 Quantized (Trained on OV3660 Sensor-Matched Dataset)
// Test Accuracy: 94.44% (100% Negative Background Suppression)
// Input: 48x48x3 RGB (INT8 quantized, scale=0.007843, zero_point=0)
// Classes (6): 0=background, 1=cream_container, 2=mouse, 3=earbuds_case, 4=glue_stick, 5=microcontroller_board
// Size: {len(bytes_data)} bytes ({len(bytes_data)/1024:.2f} KB)

alignas(16) const unsigned char g_custom_object_model_data[] = {{
"""

lines = []
for i in range(0, len(bytes_data), 12):
    chunk = bytes_data[i:i+12]
    hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
    lines.append(f"  {hex_str},\n")
source_content += "".join(lines)
source_content += f"}};\n\nconst int g_custom_object_model_data_len = {len(bytes_data)};\n"

for p in [c_header_path, fw_header_path]:
    with open(p, "w") as f:
        f.write(header_content)

for p in [c_source_path, fw_source_path]:
    with open(p, "w") as f:
        f.write(source_content)

print(f"Deployment artifacts generated successfully!")
print(f"TFLite: {dst_tflite} ({len(bytes_data)} bytes)")
print(f"Firmware arrays updated in {fw_header_path} and {fw_source_path}")

