#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from msnet import MSNet
from dataset import PotsdamSplitDataset

def load_model(checkpoint_path, num_classes, device):
    model = MSNet(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path} (Epoch {checkpoint['epoch']})")
    return model

def evaluate_model(model, dataloader, device, visualize=False, num_samples=5):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    samples_visualized = 0
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        if visualize and samples_visualized < num_samples:
            predictions = torch.argmax(outputs, dim=1)
            visualize_sample(images, masks, predictions)
            samples_visualized += 1
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Test Loss: {avg_loss:.4f}")

def visualize_sample(images, ground_truth, predictions):
    image = images[0].cpu()
    gt = ground_truth[0].cpu()
    pred = predictions[0].cpu()
    if image.shape[0] >= 3:
        rgb_image = image[:3, :, :]
    else:
        rgb_image = image
    rgb_image = rgb_image.permute(1, 2, 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate MSNet segmentation model on test dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--images-csv", type=str,
                        default="/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/test/images.csv",
                        help="Path to the test images CSV file.")
    parser.add_argument("--labels-csv", type=str,
                        default="/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/test/labels.csv",
                        help="Path to the test labels CSV file.")
    parser.add_argument("--classes-json", type=str,
                        default="/mnt/e/ISPRS-Potsdam-adri/postdam_classes.json",
                        help="Path to the classes JSON file.")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample predictions.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading test dataset...")
    test_dataset = PotsdamSplitDataset(args.images_csv, args.labels_csv, args.classes_json, scale=1.0, base_dir=None)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = len(test_dataset.color_to_index)
    model = load_model(args.checkpoint, num_classes, device)
    evaluate_model(model, test_loader, device, visualize=args.visualize, num_samples=5)

if __name__ == '__main__':
    main()
