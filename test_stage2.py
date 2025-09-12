from dataset import LiTS
from model import Unet
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from utils import *
import os
import shutil
import json
import cv2
import numpy as np  
from tqdm import tqdm

def get_args():
    parser = ArgumentParser(description='train unet')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--root', '-r', type=str, default=r'D:\DLFS\Unet\sample')
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument('--lowerbound', '-lb', type=int, default=0)
    parser.add_argument('--upperbound', '-ub', type=int, default=100)
    parser.add_argument('--json_dir', '-jd', type=str, default=None)
    parser.add_argument('--liver_mask', '-lm', type=bool, default=None)
    parser.add_argument("--bce_weight", "-bw", type=float, default=1.0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    transform = Compose([
                    ToTensor()
                ])
    
    target_transform = Compose([
                    ToTensor()
                ])
    test_dataset = LiTS(
                    root=args.root, 
                    train=False, 
                    lowerbound=args.lowerbound,
                    upperbound=args.upperbound,
                    transform=transform, 
                    target_transform=target_transform, 
                    liver_mask=args.liver_mask
                )
    
    test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False 
                )
    model = Unet()
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    all_predictions = []
    all_masks = []
    all_tumor_masks = []
    all_tumor_preds = []

    kernel = np.ones((3,3), np.uint8)

    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        image, mask, liver_mask = batch
        image = image.to(device)
        mask = mask.to(device)
        liver_mask = liver_mask.to(device)
        
        with torch.no_grad():
            pred = model(image)
            
        pred_bin = (pred > 0.5).long()
        mask_bin = mask.long()
        liver_mask_bin = liver_mask.long()
        
        tumor_pred = liver_mask_bin ^ pred_bin
        tumor_mask = liver_mask_bin ^ mask_bin
        
        prediction = pred_bin.cpu().numpy()
        mask_np = mask_bin.cpu().numpy()
        tumor_pred_np = tumor_pred.cpu().numpy()
        tumor_mask_np = tumor_mask.cpu().numpy()
        
        batch_opening = []
        for i in range(tumor_pred_np.shape[0]):  
            pred_i = tumor_pred_np[i, 0]  
            opening_i = cv2.morphologyEx(tumor_pred_np.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            batch_opening.append(opening_i)

        batch_opening = np.array(batch_opening)           
        batch_opening = np.expand_dims(batch_opening, 1)

        all_predictions.extend(prediction)
        all_masks.extend(mask_np)
        all_tumor_preds.extend(batch_opening)
        all_tumor_masks.extend(tumor_mask_np)

    scores = compute_scores(all_predictions, all_masks)
    scores_stage2 = compute_scores(all_tumor_preds, all_tumor_masks)
    results = {}

    for metric, value in scores.items():
        results[f"score_{metric}"] = value

    for metric, value in scores_stage2.items():
        results[f"score_stage2_{metric}"] = value
    output_file = os.path.join(args.json_dir, f"scores_bce_{args.bce_weight}.json")
    with open(output_file, 'w') as f:
        
        json.dump(results, f)

    print(f"Scores saved to {output_file}")