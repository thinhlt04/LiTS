import torch.nn as nn
import torch
import numpy as np

def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(2,3))  
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))  
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean() 

def bce_dice_loss(pred, target, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
    
    bce_loss = nn.BCELoss()(pred, target)
    
    intersection = (pred * target).sum(dim=(2, 3)) 
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    
    total_loss = bce_weight * bce_loss + dice_weight * dice_loss
    return total_loss

def dice(y_pred, y_true, smooth=1e-6):
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def voe(y_pred, y_true, smooth=1e-6):
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return 1 - (intersection + smooth) / (union + smooth)

def rvd(y_pred, y_true, smooth=1e-6):
    return (y_pred.sum() - y_true.sum()) / (y_true.sum() + smooth)
    
def iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou
def compute_scores(predicted_masks: list, masks: list) -> dict:
    metrics = {
        "iou": iou,
        "dice": dice
        # "voe": voe,
        # "rvd": rvd
    }
    scores = {metric_name: [] for metric_name in metrics}
    
    for predicted_mask, mask in zip(predicted_masks, masks):  
        for metric_name, scorer in metrics.items():
            scores[metric_name].append(scorer(predicted_mask, mask)) 
    
    return {metric_name: np.mean(values) for metric_name, values in scores.items()}