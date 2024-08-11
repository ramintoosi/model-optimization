"""
This module loads the CIFAR data and creates dataloaders for training and validation.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transform():
    """
    Get the transformation for the CIFAR dataset.

    :return: Transformation for the CIFAR dataset for train with augmentations and val.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }


def load_data(batch_size: int = 128, num_workers: int = 0):
    """
    Load the CIFAR dataset and create dataloaders for training and validation.

    :param batch_size: Batch size for training and validation.
    :param num_workers: Number of workers for dataloaders.
    :return: Dataloaders for training and validation.
    """

    transform = get_transform()

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform['train'])
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader}
