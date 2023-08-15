import torch
import torch.nn.functional as F
from torcheval.metrics import AUC, BinaryF1Score

from opendataval.util import FuncEnum

torch_f1_score = BinaryF1Score()
torch_f1_auc = AUC()


def accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute accuracy of two one-hot encoding tensors."""
    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()


def f1_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute f1-score of two one-hot encoding tensors."""
    return (
        torch_f1_score.update(a.argmax(dim=1), b.argmax(dim=1)).compute().float().item()
    )


def auc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute AUC of two one-hot encoding tensors."""
    return torch_f1_auc.update(a.max(dim=1), b.max(dim=1)).compute().float().item()


def neg_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return -torch.square(a - b).sum().sqrt().item()


def neg_mse(a: torch.Tensor, b: torch.Tensor):
    return -F.mse_loss(a, b).item()


def neg_l1_loss(a: torch.Tensor, b: torch.Tensor):
    return -F.l1_loss(a, b).item()


class Metrics(FuncEnum):
    ACCURACY = FuncEnum.wrap(accuracy)
    F1_SCORE = FuncEnum.wrap(f1_score)
    AUC = FuncEnum.wrap(auc)
    NEG_L2 = FuncEnum.wrap(neg_l2)
    NEG_MSE = FuncEnum.wrap(neg_mse)
    NEG_L1_LOSS = FuncEnum.wrap(neg_l1_loss)
