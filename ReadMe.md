PTQ results:
```
Original model accuracy: 0.9550, loss: 0.2713, inference time: 54.28ms
Quantized dynamic model accuracy: 0.96, loss: 0.27, inference time: 53.95ms
Quantized static model accuracy: 0.95, loss: 0.28, inference time: 22.96ms
Quantized static model with FX accuracy: 0.95, loss: 0.28, inference time: 21.37ms
```

pruning
```
Quantized static & 50% pruned: 0.93, loss: 0.19, inference time: 20.02ms
```