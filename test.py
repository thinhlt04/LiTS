from dataset import LiTS
from model import Unet
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from utils import *
import os
import shutil
import json

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

    for batch in test_loader:
        image, mask = batch
        image = image.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            pred = model(image)
            
        prediction = (pred > 0.5).long().cpu().numpy()
        mask = mask.cpu().numpy()

        all_predictions.extend(prediction)
        all_masks.extend(mask)

    scores = compute_scores(all_predictions, all_masks)
    output_file = os.path.join(args.json_dir, f"scores_bce_{args.bce_weight}.json")
    with open(output_file, 'w') as f:
        json.dump(scores, f)

    print(f"Scores saved to {output_file}")