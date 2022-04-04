import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

fer_emo_dis = [
    [0.777334, 0.03261, 0.017447, 0.114091, 0.026224, 0.007145, 0.004379, 0.020772],
    [0.042842, 0.917618, 0.023221, 0.004816, 0.004217, 0.001969, 0.001755, 0.003563],
    [0.058354, 0.035775, 0.787254, 0.01037, 0.014319, 0.003015, 0.088513, 0.0024],
    [0.170124, 0.013292, 0.008632, 0.747902, 0.022808, 0.011237, 0.015923, 0.010082],
    [0.074781, 0.02535, 0.060215, 0.031924, 0.740551, 0.034938, 0.023251, 0.008989],
    [0.08959, 0.017445, 0.016911, 0.088629, 0.125152, 0.620813, 0.007788, 0.033671],
    [0.042815, 0.012239, 0.169483, 0.072557, 0.027228, 0.013045, 0.658964, 0.003671],
    [0.147933, 0.022174, 0.00989, 0.042923, 0.033551, 0.059924, 0.005507, 0.678099]
]

class LabelSmoothingLoss(torch.nn.Module):
    """
    For Label Smoothing loss (not one-hot but softer) with alpha default 0.9.
    Similar to `prlab.fastai.utils.LabelSmoothingLoss1` but fix the hs of 1-alpha part and safer i_mtx
    pred: [bs, lbl_size], target: [bs]
    """

    def __init__(self, alpha=0.9, gamma=2, reduction='mean', **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        lbl_size = pred.size()[-1]
        i_mtx = torch.eye(lbl_size).to(target.device)
        t = torch.embedding(i_mtx, target)
        soft_part = torch.ones_like(t)
        l_softmax = log_softmax(pred, 1)
        out = -t * l_softmax * self.alpha - soft_part * l_softmax * (1 - self.alpha) / lbl_size
        out = out.sum(dim=-1)

        return do_reduction(out, reduction=self.reduction)

class LabelSmoothingLossDis(LabelSmoothingLoss):
    """
    Like `LabelSmoothingLoss`, but the given is distribution (by statistical)
    pred: [bs, lbl_size], target: [bs, lbl_size]
    """

    def __init__(self, alpha=0.5, gamma=2, **kwargs):
        super().__init__(alpha=alpha, gamma=gamma, **kwargs)

    def do_reduction(self, loss_tensor, reduction='mean'):
        """
        reduction that use in most loss function
        :param loss_tensor:
        :param reduction: str => none (None), mean, sum
        :return: out with reduction
        """
        if isinstance(reduction, str) and hasattr(loss_tensor, reduction):
            loss_tensor = getattr(loss_tensor, reduction)()
        return loss_tensor

    def forward(self, pred, target):
        target_dev = target.device
        lbl_size = pred.size()[-1]
        target = [fer_emo_dis[i] for i in target]
        target = torch.tensor(target).to(target_dev)
        i_mtx = torch.eye(lbl_size).to(target_dev)

        with torch.no_grad():
            lbl_correct = target.argmax(dim=-1)
            one_hot = torch.embedding(i_mtx, lbl_correct)

        l_softmax = log_softmax(pred, 1)
        out = -(one_hot * self.alpha + target * (1 - self.alpha)) * l_softmax
        out = out.sum(dim=-1)

        return self.do_reduction(out, reduction=self.reduction)

class FocalLabelSmoothingLossDis(LabelSmoothingLossDis):
    """
    Like `LabelSmoothingLossDis`, but focal loss is combined additionaly
    pred: [bs, lbl_size], target: [bs, lbl_size]
    """

    def __init__(self, alpha=0.8, gamma=2, **kwargs):
        super().__init__(alpha=alpha, gamma=gamma, **kwargs)

    def do_reduction(self, loss_tensor, reduction='mean'):
        """
        reduction that use in most loss function
        :param loss_tensor:
        :param reduction: str => none (None), mean, sum
        :return: out with reduction
        """
        if isinstance(reduction, str) and hasattr(loss_tensor, reduction):
            loss_tensor = getattr(loss_tensor, reduction)()
        return loss_tensor

    def forward(self, pred, target):
        target_dev = target.device
        lbl_size = pred.size()[-1]
        target = [fer_emo_dis[i] for i in target]
        target = torch.tensor(target).to(target_dev)
        i_mtx = torch.eye(lbl_size).to(target_dev)

        with torch.no_grad():
            lbl_correct = target.argmax(dim=-1)
            one_hot = torch.embedding(i_mtx, lbl_correct)

        # Focal PDLS
        l_softmax = log_softmax(pred, 1)
        pt = torch.exp(l_softmax)
        out = -(one_hot * self.alpha + target * (1 - self.alpha)) * l_softmax * (1 - pt) ** self.gamma 
        out = out.sum(dim=-1)

        return self.do_reduction(out, reduction=self.reduction)

class CompFocalLabelSmoothingLossDis(LabelSmoothingLossDis):
    """
    Like `LabelSmoothingLossDis`, but focal loss is combined additionaly
    pred: [bs, lbl_size], target: [bs, lbl_size]
    """

    def __init__(self, alpha=0.5, gamma=2, **kwargs):
        super().__init__(alpha=alpha, gamma=gamma, **kwargs)

    # Loss functions/class
    def do_reduction(self, loss_tensor, reduction='mean'):
        """
        reduction that use in most loss function
        :param loss_tensor:
        :param reduction: str => none (None), mean, sum
        :return: out with reduction
        """
        if isinstance(reduction, str) and hasattr(loss_tensor, reduction):
            loss_tensor = getattr(loss_tensor, reduction)()
        return loss_tensor

    def forward(self, pred, target):
        target_dev = target.device
        lbl_size = pred.size()[-1]
        target = [fer_emo_dis[i] for i in target]
        target = torch.tensor(target).to(target_dev)
        i_mtx = torch.eye(lbl_size).to(target_dev)

        with torch.no_grad():
            lbl_correct = target.argmax(dim=-1)
            one_hot = torch.embedding(i_mtx, lbl_correct)

        # Compensated Focal PDLS
        l_softmax = log_softmax(pred, 1)
        pt = torch.exp(l_softmax)
        out = -(one_hot * self.alpha + target * (1 - self.alpha)) * l_softmax * (1 - pt) * (lbl_size / (lbl_size -1)) 
        out = out.sum(dim=-1)

        return self.do_reduction(out, reduction=self.reduction)

