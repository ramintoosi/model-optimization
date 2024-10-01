# ResNet Quantization and Pruning

This project demonstrates the process of quantization and pruning on a ResNet model using PyTorch. The goal is to reduce the model size and improve inference time while maintaining accuracy.
However, the project is only a simple implementation of the quantization and pruning techniques on a ResNet model for educational purposes. 

[See my blog post for full description](https://ramintoosi.ir/posts/2024/10/blog-post-1/).

---


## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Training](#training)
- [Quantization](#quantization)
- [Pruning](#pruning)
- [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
- [Results](#results)
- [References](#references)

## Introduction
Quantization and pruning are techniques used to optimize deep learning models for deployment on resource-constrained devices. This project applies these techniques to a ResNet model to achieve efficient inference.

## Setup
1. **Clone the repository**:
    ```sh
    git clone https://github.com/ramintoosi/resnet-quantization-pruning.git
    cd resnet-quantization-pruning
    ```

2. **Install dependencies**:
    ```sh
    pip install torch torchvision torchaudio
    ```
   or
    ```sh
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

## Training
To train the ResNet model from scratch, run:
```sh
python train_model_simple.py
```
This will train the model on the CIFAR-10 
dataset and save the trained model as `weights/simple_best_model.pth`.

## Quantization
### Post-Training Quantization (PTQ)
To apply post-training quantization, run:
```sh
python post_train_quant.py
```

### Quantization-Aware Training (QAT)
To train the model with quantization-aware training, run:
```sh
python train_model_QAT.py
```

## Pruning
To prune the model, run:
```sh
python post_training_pruning.py
```
This code will also use quantization to further optimize the pruned model.

## Quantization-Aware Training (QAT) with Pruning
To train the model with both QAT and pruning, run:
```sh
python train_model_QAT_prune.py
```

## Results
Results were obtained using an RTX 2080 Ti.

| Model Type               | Accuracy | Loss | Inference Time  |
|--------------------------|----------|------|-----------------|
| Original model           | 0.95     | 0.27 | 54.28ms         |
| PTQ dynamic model        | 0.96     | 0.27 | 53.95ms         |
| PTQ static model         | 0.95     | 0.28 | 22.96ms         |
| PTQ static model with FX | 0.95     | 0.28 | 21.37ms         |
| Pruned 50% + PTQ static  | 0.93     | 0.19 | 20.02ms         |
| QAT                      | 0.95     | 0.27 | 19.87ms         |
| Pruned 50% + QAT         | 0.95     | 0.25 | 20.61ms         |

## References
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
