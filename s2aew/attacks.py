import copy

import torch
import torch.nn.utils.prune as prune


def magnitude_pruned_copy(model: torch.nn.Module, amount: float = 0.1) -> torch.nn.Module:
    """Return a copied model after unstructured magnitude pruning of Conv/Linear weights."""
    pruned = copy.deepcopy(model)
    for module in pruned.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return pruned


def freeze_except_classifier(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze feature layers and keep common classifier heads trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if any(token in name.lower() for token in ("fc", "classifier", "head")):
            parameter.requires_grad_(True)
    return model
