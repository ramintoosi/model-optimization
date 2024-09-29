from torch import nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
import torch.ao.quantization.quantize_fx as quantize_fx
import copy



def prepare_model_qat(model_fp: nn.Module, example_inputs):
    model_to_quantize = copy.deepcopy(model_fp)
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    model_to_quantize.train()
    # prepare
    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

    return model_prepared