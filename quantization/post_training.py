"""
This module implements post-training quantization of a PyTorch model.
"""
import copy

import torch
from torch.ao import quantization as quan
import torch.ao.quantization.quantize_fx as quantize_fx
from data import load_data
from tqdm import tqdm


# Dynamic Quantization: This method quantizes only the activations during inference, while weights
# are quantized beforehand. This means that the quantization overhead occurs during the forward pass, but
# since it happens on-the-fly, there's no additional pre-processing step needed.
# Therefore, inference time remains unaffected.
# This is used for situations where the model execution time is dominated by loading weights
# from memory rather than computing the matrix multiplications.
def quantize_dynamic(model_fp32: torch.nn.Module, dtype=torch.qint8):
    """
    Quantize a PyTorch model using dynamic quantization.
    :param model_fp32: model to quantize
    :param dtype: target dtype for quantized weights
    :return: quantized model
    """
    # create a quantized model instance
    model_quantized = torch.ao.quantization.quantize_dynamic(
        model_fp32,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=dtype)  # the target dtype for quantized weights
    return model_quantized


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.model = model

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def quantize_static(model: torch.nn.Module):
    """
    Quantize a PyTorch model using static quantization.
    :param model: model to quantize
    :return: quantized model
    """
    # wrap the model with the ModelWrapper to include the quant and dequant stubs
    model_fp32 = ModelWrapper(model)

    # model must be set to eval mode for static quantization logic to work

    model_fp32.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'x86' for server inference and 'qnnpack'
    # for mobile inference. Other quantization configurations such as selecting
    # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
    # can be specified here.
    qconfig = quan.get_default_qconfig('x86')
    model_fp32.qconfig = qconfig

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    # instead of an empty dataset.
    dataloader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fp32_prepared.to(device)
    with torch.no_grad():
        for data, _ in tqdm(dataloader['train'], desc='Calibrating model', unit=' batch'):
            model_fp32_prepared(data.to(device))
    model_fp32_prepared.cpu()
    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    return model_int8


def quantize_static_fx(model_fp: torch.nn.Module):
    """
    Quantize a PyTorch model using static quantization with FX graph mode.
    :param model_fp: model to quantize
    :return: quantized model
    """
    model_to_quantize = copy.deepcopy(model_fp)
    qconfig_mapping = quan.get_default_qconfig_mapping("x86")
    model_to_quantize.eval()
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, (torch.rand((1, 3, 224, 224)),))
    # calibrate
    dataloader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prepared.to(device)
    with torch.no_grad():
        for data, _ in tqdm(dataloader['train'], desc='Calibrating model FX', unit=' batch'):
            model_prepared(data.to(device))
    model_prepared.cpu()
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)

    return model_quantized
