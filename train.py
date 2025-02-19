#!/usr/bin/env python
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from msnet import MSNet
from dataset import PotsdamSplitDataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    progress = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, batch in progress:
        images = batch['image'].to(device)
        assert images.shape[1] == 4, f"Batch images have {images.shape[1]} channels, expected 4."
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    epoch_time = time.time() - epoch_start_time
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, epoch_time

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    progress = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Validating", leave=False)
    with torch.no_grad():
        for batch_idx, batch in progress:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MSNet segmentation model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--images-csv", type=str,
                        default="/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/train/images.csv",
                        help="Path to the images CSV file.")
    parser.add_argument("--labels-csv", type=str,
                        default="/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/train/labels.csv",
                        help="Path to the labels CSV file.")
    parser.add_argument("--classes-json", type=str,
                        default="/mnt/e/ISPRS-Potsdam-adri/postdam_classes.json",
                        help="Path to the classes JSON file.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("Starting training setup...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading dataset...")
    dataset = PotsdamSplitDataset(args.images_csv, args.labels_csv, args.classes_json, scale=1.0, base_dir=None)
    print(f"Dataset loaded with {len(dataset)} samples.")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split dataset: {train_size} training samples, {val_size} validation samples.")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = len(dataset.color_to_index)
    model = MSNet(num_classes=num_classes).to(device)
    print(f"Initialized model with {num_classes} classes.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Beginning training loop...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} - Starting training...")
        train_loss, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")
        print(f"Epoch {epoch+1} - Starting validation...")
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()
