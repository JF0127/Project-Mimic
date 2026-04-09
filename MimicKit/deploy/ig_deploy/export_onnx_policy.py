
import torch
import os

dest_path = "/home/wyh/projects/MimicKit/output"
policy_jit_path = os.path.join(dest_path, 'model_jit.pt')
policy_jit_model = torch.jit.load(policy_jit_path)
policy_jit_model.eval()

# 核心修改：MimicKit 的观测维度是 206
test_input_tensor = torch.randn(1, 206)

policy_onnx_model = dest_path + '/model.onnx'

torch.onnx.export(policy_jit_model,
                  test_input_tensor,
                  policy_onnx_model,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['obs'],
                  output_names=['action'],
                  )

print(f"ONNX 模型导出成功: {policy_onnx_model}")