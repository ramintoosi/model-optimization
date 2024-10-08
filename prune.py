import torch
import torch.nn.utils.prune as prune

def make_sparse(model_to_prune, rate=0.5):
    """
    This function prunes the model by making the weights sparse.
    :param model_to_prune: model to prune
    :param rate: the percentage of weights to prune
    """
    for name, module in model_to_prune.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=rate)
            prune.remove(module, 'weight')
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=rate)
            prune.remove(module, 'weight')
