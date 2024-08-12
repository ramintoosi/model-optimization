"""
This module validates and calculates the accuracy of a model on MNIST validation data.
"""
import time

import torch
from tqdm import tqdm

from data import load_data


def validate(model, device):
    """
    Validate the model on the validation data.

    :param device: cuda or cpu.
    :param model: Model to validate.
    :return: Tuple of accuracy, loss, and average inference time (ms).
    """
    dataloaders = load_data(batch_size=1, num_workers=0)
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    running_loss = 0.0
    start_time = time.time()
    i_data = 0
    n_total = 2000
    with torch.no_grad():
        for data in tqdm(dataloaders['val'], total=n_total, desc='Validating model', unit=' image'):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

            i_data += 1

            if i_data > n_total:
                break

    elapsed_time = time.time() - start_time
    accuracy = correct / total
    loss = running_loss / total
    avg_inference_time = elapsed_time / total
    return accuracy, loss, avg_inference_time * 1000

