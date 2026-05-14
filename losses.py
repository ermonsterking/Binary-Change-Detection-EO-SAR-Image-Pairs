%%writefile losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        tgts = targets.view(-1).float()
        intersection = (probs * tgts).sum()
        return 1 - (2. * intersection + self.smooth) / (probs.sum() + tgts.sum() + self.smooth)

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=0.75, gamma=2.0):
        super().__init__()
        self.dw, self.fw = dice_weight, focal_weight
        self.dice = DiceLoss()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        logits = logits.squeeze(1) # [B, 1, H, W] -> [B, H, W]
        # BCE with logit flattening
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="mean")
        dice = self.dice(logits, targets)
        return (self.dw * dice) + (self.fw * bce)

def build_loss(cfg):
    return CombinedLoss(dice_weight=cfg['dice_weight'], focal_weight=cfg['focal_weight'])
