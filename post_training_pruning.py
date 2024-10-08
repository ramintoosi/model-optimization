"""
This module quantizes a PyTorch model using post-training quantization and pruning.
"""
import copy
from os.path import isfile

import torch

from validation import validate
from model.resnet import get_model
from quantization.post_training import quantize_static_fx
from prune import make_sparse


model = get_model(num_classes=10)
checkpoint = "weights/original_model.pt"
model.load_state_dict(torch.load(checkpoint))

model_orig = copy.deepcopy(model)

checkpoint_quantized_prune = "weights/quantized_prune_model.pt"
if isfile(checkpoint_quantized_prune):
    model_quantized_prune = torch.jit.load(checkpoint_quantized_prune)
else:
    make_sparse(model)
    model_quantized_prune = quantize_static_fx(model)
    traced = torch.jit.trace(model_quantized_prune, torch.rand((1, 3, 224, 224)))
    torch.jit.save(traced, checkpoint_quantized_prune)

# validate models
device = torch.device("cpu")
accuracy, loss, inference_time = validate(model_orig, device, n_total=100)
accuracy_quantized, loss_quantized, inference_time_quantized = validate(model_quantized_prune, device, n_total=100)

# print the results
print(f"Original model accuracy: {accuracy:.4f}, loss: {loss:.4f}, inference time: {inference_time:.2f}ms")
print(f"Quantized static model accuracy: {accuracy_quantized:.2f}, loss: {loss_quantized:.2f}, "
      f"inference time: {inference_time_quantized:.2f}ms")
