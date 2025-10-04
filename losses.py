#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss
    
class WeightedCrossEntropy:
    def __init__(self, **kwargs):
        import torch
        self.idk = kwargs['idk']
        weights = kwargs.get('weights', [1.0, 2.0, 1.5, 1.0, 1.0]) # for the different classes, this performed best
        self.weights = torch.tensor(weights)
        print(f"Initialized {self.__class__.__name__} with weights: {weights}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        weights = self.weights.to(pred_softmax.device)

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()
        
        weighted_mask = mask * weights[None, :, None, None]

        loss = - einsum("bkwh,bkwh->", weighted_mask, log_p)
        loss /= weighted_mask.sum() + 1e-10

        return loss

class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs.get('idk', None)
        self.smooth = kwargs.get('smooth', 1e-7)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        if self.idk is not None:
            pred_softmax = pred_softmax[:, self.idk, ...]
            weak_target = weak_target[:, self.idk, ...]

        weak_target = weak_target.float()

        intersection = einsum("bkwh,bkwh->bk", pred_softmax, weak_target)
        pred_sum = einsum("bkwh->bk", pred_softmax)
        target_sum = einsum("bkwh->bk", weak_target)

        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return 1 - dice.mean()

class CeAndDiceCombinedLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs.get('idk', None)
        self.ce_weight = kwargs.get('ce_weight', 1.0)
        self.dice_weight = kwargs.get('dice_weight', 1.0)
        
        self.ce_loss = CrossEntropy(idk=self.idk)
        self.dice_loss = DiceLoss(idk=self.idk)
        
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        ce = self.ce_loss(pred_softmax, weak_target)
        dice = self.dice_loss(pred_softmax, weak_target)
        
        return self.ce_weight * ce + self.dice_weight * dice

class FocalTverskyLoss:
    def __init__(self, **kwargs):
        self.idk = kwargs.get('idk', [0, 1, 2, 3, 4])
        self.alpha = kwargs.get('alpha', 0.7)
        self.beta = kwargs.get('beta', 0.3)
        self.gamma = kwargs.get('gamma', 0.75)
        self.smooth = kwargs.get('smooth', 1e-6)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target_one_hot):
        assert pred_softmax.shape == target_one_hot.shape
        assert simplex(pred_softmax)
        assert sset(target_one_hot, [0, 1])
        
        if self.idk is not None:
            pred_softmax = pred_softmax[:, self.idk, ...]
            target_one_hot = target_one_hot[:, self.idk, ...]
        
        target_one_hot = target_one_hot.float()
        
        TP = einsum("bkwh,bkwh->bk", pred_softmax, target_one_hot)
        FP = einsum("bkwh,bkwh->bk", pred_softmax, 1 - target_one_hot)
        FN = einsum("bkwh,bkwh->bk", 1 - pred_softmax, target_one_hot)
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        focal_tversky = ((1 - tversky) ** self.gamma).mean()
        
        return focal_tversky
    
class TverskyLoss:
    def __init__(self, **kwargs):
        self.idk = kwargs.get('idk', [0, 1, 2, 3, 4])
        self.alpha = kwargs.get('alpha', 0.7)
        self.beta = kwargs.get('beta', 0.3)
        self.smooth = kwargs.get('smooth', 1.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target_one_hot):
        assert pred_softmax.shape == target_one_hot.shape
        assert simplex(pred_softmax)
        assert sset(target_one_hot, [0, 1])
        
        if self.idk is not None:
            pred_softmax = pred_softmax[:, self.idk, ...]
            target_one_hot = target_one_hot[:, self.idk, ...]
        
        target_one_hot = target_one_hot.float()
        
        TP = einsum("bkwh,bkwh->bk", pred_softmax, target_one_hot)
        FP = einsum("bkwh,bkwh->bk", pred_softmax, 1 - target_one_hot)
        FN = einsum("bkwh,bkwh->bk", 1 - pred_softmax, target_one_hot)
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1.0 - tversky.mean()
    
class CeTverskyCombo:
    def __init__(self, **kwargs):
        self.idk = kwargs.get('idk', [0, 1, 2, 3, 4])
        self.ce_weight = kwargs.get('ce_weight', 0.5)
        self.tversky_weight = kwargs.get('tversky_weight', 1.0)
        
        self.ce_loss = CrossEntropy(idk=self.idk)
        self.tversky_loss = TverskyLoss(idk=self.idk, alpha=0.7, beta=0.3, smooth=1.0)
        
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        ce = self.ce_loss(pred_softmax, weak_target)
        tversky = self.tversky_loss(pred_softmax, weak_target)
        
        return self.ce_weight * ce + self.tversky_weight * tversky
    
class FocalDiceLoss:
    def __init__(self, **kwargs):
        self.idk = kwargs.get('idk', [0, 1, 2, 3, 4])
        self.gamma = kwargs.get('gamma', 2.0)
        self.alpha = kwargs.get('alpha', 0.25)
        self.dice_weight = kwargs.get('dice_weight', 1.0)
        self.focal_weight = kwargs.get('focal_weight', 1.0)
        print(f"Initialized {self.__class__.__name__} with gamma={self.gamma}, focal_weight={self.focal_weight}, dice_weight={self.dice_weight}")

    def __call__(self, pred_softmax, target):
        pred = pred_softmax[:, self.idk, ...]
        tgt = target[:, self.idk, ...].float()

        ce = -(tgt * (pred + 1e-10).log())
        focal_weight = (1 - pred) ** self.gamma
        focal_loss = (self.alpha * focal_weight * ce).sum() / (tgt.sum() + 1e-10)
        
        intersection = einsum("bk...,bk...->bk", pred, tgt)
        union = einsum("bk...->bk", pred) + einsum("bk...->bk", tgt)
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice.mean()
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss