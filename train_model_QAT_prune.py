"""
this module is used to train the model with QAT and pruning
"""
import os


import torch
import torch.ao.quantization.quantize_fx as quantize_fx

from data import load_data
from model.resnet import get_model
from train import train
from quantization.qat import prepare_model_qat
from validation import validate
from prune import make_sparse


def train_model_qat_prune(resume=True):
    """
    Train a simple model without quantization and pruning.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloaders = load_data(batch_size=128, num_workers=0)

    model = get_model(num_classes=10)
    model.load_state_dict(torch.load("weights/original_model.pt", weights_only=True))

    make_sparse(model)

    model_prepared = prepare_model_qat(model, example_inputs = next(iter(dataloaders['train']))[0])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

    train(model_prepared, dataloaders, optimizer, criterion, scheduler,
          device, "qat_prune", 20, resume=resume)


if __name__ == '__main__':
    train_model_qat_prune(resume=False)

    checkpoint = 'weights/qat_prune_model.pt'
    if os.path.isfile(checkpoint):
        model_quantized = torch.jit.load(checkpoint)
    else:

        dataloaders = load_data(batch_size=128, num_workers=0)

        model = get_model(num_classes=10)

        make_sparse(model)

        model_prepared = prepare_model_qat(model, example_inputs=next(iter(dataloaders['train']))[0])
        model_prepared.load_state_dict(torch.load("weights/qat_prune_best_model.pt", weights_only=True)["model_state_dict"])

        model_prepared.eval()
        model_quantized = quantize_fx.convert_fx(model_prepared)

        traced = torch.jit.trace(model_quantized, torch.rand((1, 3, 224, 224)))
        torch.jit.save(traced, checkpoint)

    # validation
    device = torch.device("cpu")
    accuracy, loss, inference_time = validate(model_quantized, device)
    print(f"Quantized model (QAT) accuracy: {accuracy:.2f}, loss: {loss:.2f}, inference time: {inference_time:.2f}ms")

