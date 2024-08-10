"""
this module is used to train a simple model without quantization and pruning
"""

import torch
from data import load_data
from model import get_resnet18
from train import train


def train_model_simple():
    """
    Train a simple model without quantization and pruning.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloaders = load_data(batch_size=128, num_workers=0)

    model = get_resnet18(num_classes=10)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

    train(model, dataloaders, optimizer, criterion, scheduler, device, "simple", 25)


if __name__ == '__main__':
    train_model_simple()