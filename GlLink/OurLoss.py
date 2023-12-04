from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)
    sh = list(labels.shape)
    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)
    return labels

class OurLoss(_Loss):
    def __init__(self,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch:
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))
        pt = (input * target).sum(dim=1)
        new_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            result = torch.mean(new_loss)
        elif self.reduction == 'sum':
            result = torch.sum(new_loss)
        elif self.reduction == 'none':
            result = new_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return result

