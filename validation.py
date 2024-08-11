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

    :param model: Model to validate.
    :param device: Device to run the model on.
    :return: Tuple of accuracy, loss, and average inference time (ms).
    """
    dataloaders = load_data(batch_size=1, num_workers=0)
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    start_time = time.time()
    i_data = 0
    with torch.no_grad():
        for data in tqdm(dataloaders['val'], total=2000):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

            i_data += 1

            if i_data > 2000:
                break

    elapsed_time = time.time() - start_time
    accuracy = correct / total
    loss = running_loss / total
    avg_inference_time = elapsed_time / total
    return accuracy, loss, avg_inference_time * 1000


def recall_precision(model, device):
    """
    Calculate the recall and precision of the model on the validation data.

    :param model: Model to validate.
    :param device: Device to run the model on.
    :return: Tuple of recall and precision.
    """
    dataloaders = load_data(batch_size=128, num_workers=0)
    model.eval()
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for data in dataloaders['val']:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    return recall, precision
