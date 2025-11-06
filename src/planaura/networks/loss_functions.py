import torch
import torch.nn as nn


class IgnoreLoss(nn.Module):

    def __init__(self):
        super(IgnoreLoss, self).__init__()

    def forward(self, predicted_im, target_im, mask=None):
        loss = torch.tensor(0).float().to(device=predicted_im.device)
        return loss


class SimpleLoss(nn.Module):

    def __init__(self, no_data=None):
        super(SimpleLoss, self).__init__()
        self.no_data = no_data

    def forward(self, pred, target, mask):
        if self.no_data is not None:
            mask_no_data = target != self.no_data
            pred = pred * mask_no_data
            target = target * mask_no_data
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        sum_mask = mask.sum()
        if sum_mask != 0:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = (loss * mask).sum()
        return loss
