import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import onnx
import subprocess
import os
import shutil

class TinyTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = TinyTestModel()
model.eval()
dummy_input = torch.randn(1, 3, 48, 48)

test_dir = r"C:\All_Projects\esp32_cam_ai_system\models\candidate_models\test_export"
os.makedirs(test_dir, exist_ok=True)
onnx_path = os.path.join(test_dir, "test_model.onnx")

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)
print("ONNX export succeeded:", os.path.exists(onnx_path), os.path.getsize(onnx_path), "bytes")

# Run onnx2tf to convert to TFLite
cmd = [
    "python", "-m", "onnx2tf",
    "-i", onnx_path,
    "-o", os.path.join(test_dir, "tflite_out"),
    "-tb", "flatbuffer_direct"
]
print("Running onnx2tf conversion...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("onnx2tf returncode:", res.returncode)
print("Stdout:", res.stdout[-400:])
if res.stderr:
    print("Stderr:", res.stderr[-400:])
tflite_path = os.path.join(test_dir, "tflite_out", "test_model_float32.tflite")
if os.path.exists(tflite_path):
    print("TFLite float32 model created! Size:", os.path.getsize(tflite_path), "bytes")

