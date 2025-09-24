from model import Unet
from dataset import *
import torch
from torch.optim import Adam
from utils import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import shutil


def get_args():
    parser = ArgumentParser(description="train unet")
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--batch_size", "-b", type=int, default=2)
    parser.add_argument("--root", "-r", type=str, default=r"D:\DLFS\Unet\sample")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--lowerbound", "-lb", type=int, default=0)
    parser.add_argument("--upperbound", "-ub", type=int, default=100)
    parser.add_argument("--bce_weight", "-bw", type=float, default=1.0)
    parser.add_argument("--dice_weight", "-dw", type=float, default=1.0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    transform = Compose([ToTensor()])

    target_transform = Compose([ToTensor()])

    train_dataset = LiTS_stage2(
        root=args.root,
        train=True,
        lowerbound=args.lowerbound,
        upperbound=args.upperbound,
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    dev_dataset = LiTS_stage2(
        root=args.root,
        dev=True,
        lowerbound=args.lowerbound,
        upperbound=args.upperbound,
        transform=transform,
        target_transform=target_transform
    )

    dev_loader = DataLoader(
        dataset=dev_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Unet().to(device)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    writer = SummaryWriter(args.logging)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

    num_iters = len(train_loader)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start_epoch = checkpoint["epoch"]
        best_iou = checkpoint["best_iou"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        start_epoch = 0
        best_iou = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        process_bar = tqdm(train_loader, colour="cyan")
        for iter, (input, masked_target, liver_mask, target) in enumerate(process_bar):
            if torch.cuda.is_available():
                input = input.to(device)
                masked_target = masked_target.to(device)
 
            # forward
            outputs = model(input)
            loss_value = bce_dice_loss(
                outputs, masked_target, bce_weight=args.bce_weight, dice_weight=args.dice_weight
            )
            process_bar.set_description(
                f"Epochs {epoch + 1}/{args.epochs}. Iteration {iter + 1}/{num_iters}. Loss {loss_value:.3f}"
            )
            writer.add_scalar("Train/Loss", loss_value, epoch * num_iters + iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_masks = []

        for batch in dev_loader:
            input, masked_target, liver_mask, target = batch
            input = input.to(device)
            masked_target = masked_target.to(device)

            with torch.no_grad():
                pred = model(input)

            prediction = (pred > 0.5).long().cpu().numpy()
            masked_target = masked_target.cpu().numpy()

            all_predictions.extend(prediction)
            all_masks.extend(masked_target)
        score = compute_scores(all_predictions, all_masks)
        iou_score = score["iou"]
        writer.add_scalar("Val/IoU", iou_score, epoch)
        scheduler.step(iou_score)
        checkpoint = {
            "epoch": epoch + 1,
            "best_iou": best_iou,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{args.trained_models}/last_model.pt")
        if iou_score > best_iou:
            checkpoint = {
                "epoch": epoch + 1,
                "best_iou": iou_score,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{args.trained_models}/best_model.pt")
            best_iou = iou_score
