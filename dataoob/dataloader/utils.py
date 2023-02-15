import torch
import torch.nn.functional as F


def one_hot_encode(*tensors: tuple[torch.tensor]) -> torch.tensor:
    label_dim = torch.max(torch.concat(tensors)) + 1
    one_hotify = lambda t: F.one_hot(t.long(), num_classes=int(label_dim)).float()
    return (one_hotify(t) for t in tensors)