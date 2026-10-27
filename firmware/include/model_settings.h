#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

// DeskObjectNet Model Dimensions (INT8 48x48x3 RGB)
constexpr int kNumCols = 48;
constexpr int kNumRows = 48;
constexpr int kNumChannels = 3;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 6;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // MODEL_SETTINGS_H_
