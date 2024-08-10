"""
This module loads the MNIST data and creates dataloaders for training and validation.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transform():
    """
    Get the transformation for the MNIST dataset.

    :return: Transformation for the MNIST dataset for train with augmentations and val.
    """
    return {
        'train': transforms.Compose([
            # make sure we have three channels
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }


def load_data(batch_size: int = 32, num_workers: int = 0):
    """
    Load the MNIST dataset and create dataloaders for training and validation.

    :param batch_size: Batch size for training and validation.
    :param num_workers: Number of workers for dataloaders.
    :return: Dataloaders for training and validation.
    """

    transform = get_transform()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform['train'])
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader}
