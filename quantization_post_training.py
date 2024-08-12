"""
This module quantizes a PyTorch model using post-training quantization.
"""
from os.path import isfile

import torch

from model.resnet import get_model
from quantization.post_training import quantize_dynamic, quantize_static, quantization_static_fx
from validation import validate

model = get_model(num_classes=10)
checkpoint = "weights/original_model.pt"
if isfile(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
else:
    model.load_state_dict(torch.load("weights/simple_best_model.pt")["model_state_dict"])
    torch.save(model.state_dict(), "weights/original_model.pt")

model_quantized = quantize_dynamic(model, dtype=torch.qint8)
checkpoint_quantized = "weights/quantized_dynamic_model.pt"
if not isfile(checkpoint_quantized):
    torch.save(model_quantized.state_dict(), checkpoint_quantized)

checkpoint_quantized_static = "weights/quantized_static_model.pt"
if isfile(checkpoint_quantized_static):
    model_quantized_static = torch.jit.load(checkpoint_quantized_static)
else:
    model_quantized_static = quantize_static(model)
    traced = torch.jit.trace(model_quantized_static, torch.rand((1, 3, 224, 224)))
    torch.jit.save(traced, checkpoint_quantized_static)

checkpoint_quantized_static_fx = "weights/quantized_static_fx_model.pt"
if isfile(checkpoint_quantized_static_fx):
    model_quantized_static_fx = torch.jit.load(checkpoint_quantized_static_fx)
else:
    model_quantized_static_fx = quantize_static(model)
    traced = torch.jit.trace(model_quantized_static_fx, torch.rand((1, 3, 224, 224)))
    torch.jit.save(traced, checkpoint_quantized_static_fx)

# device
device = torch.device("cpu")

# validate models
accuracy, loss, inference_time = validate(model, device)
accuracy_quantized, loss_quantized, inference_time_quantized = validate(model_quantized, device)
accuracy_quantized_static, loss_quantized_static, inference_time_quantized_static = (
    validate(model_quantized_static, device))
accuracy_quantized_static_fx, loss_quantized_static_fx, inference_time_quantized_static_fx = (
    validate(model_quantized_static_fx, device))

# print the results
print(f"Original model accuracy: {accuracy:.4f}, loss: {loss:.4f}, inference time: {inference_time:.2f}ms")
print(f"Quantized dynamic model accuracy: {accuracy_quantized:.2f}, loss: {loss_quantized:.2f}, "
      f"inference time: {inference_time_quantized:.2f}ms")
print(f"Quantized static model accuracy: {accuracy_quantized_static:.2f}, loss: {loss_quantized_static:.2f}, "
      f"inference time: {inference_time_quantized_static:.2f}ms")
print(f"Quantized static model with FX accuracy: {accuracy_quantized_static_fx:.2f}, "
      f"loss: {loss_quantized_static_fx:.2f}, inference time: {inference_time_quantized_static_fx:.2f}ms")
