# dataset.py
import os
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_image(filename):
    return Image.open(filename)

def get_paths_from_csv(file_path):
    # Reads the CSV and returns a list of paths (assuming the path is in the first column)
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        paths = [Path(row[0]) for row in reader]
    return paths

def mask_to_class(mask, color_to_index):
    # mask: array (H, W, 3)
    # color_to_index: dict mapping (R, G, B) to class index
    H, W, _ = mask.shape
    class_mask = np.zeros((H, W), dtype=np.int64)
    for color, idx in color_to_index.items():
        matches = np.all(mask == np.array(color), axis=-1)
        class_mask[matches] = idx
    return class_mask

class PotsdamSplitDataset(Dataset):
    def __init__(self, images_csv: str, labels_csv: str, classes_json: str, scale: float = 1.0, base_dir: str = None):
        """
        Parameters:
          - images_csv: path to the images CSV (e.g., .../split_postdam_ir_512/train/images.csv)
          - labels_csv: path to the masks CSV (e.g., .../split_postdam_ir_512/train/labels.csv)
          - classes_json: path to the postdam_classes.json file
          - scale: scaling factor for resizing images (use 1.0 if already 512x512)
          - base_dir: (optional) base directory for relative paths in the CSV files
        """
        self.scale = scale
        self.base_dir = Path(base_dir) if base_dir is not None else None

        # Read paths from the CSVs
        self.images_list = get_paths_from_csv(images_csv)
        self.labels_list = get_paths_from_csv(labels_csv)
        if len(self.images_list) != len(self.labels_list):
            raise ValueError("The number of images and masks does not match.")

        if self.base_dir is not None:
            self.images_list = [self.base_dir / p for p in self.images_list]
            self.labels_list = [self.base_dir / p for p in self.labels_list]

        with open(classes_json, 'r') as f:
            classes = json.load(f)
        self.color_to_index = {tuple(color): idx for idx, (class_name, color) in enumerate(classes.items())}

        # For debugging: print number of items
        print(f"PotsdamSplitDataset initialized with {len(self.images_list)} samples.")

    def preprocess_image(self, pil_img, is_mask=False):
        w, h = pil_img.size
        newW, newH = int(w * self.scale), int(h * self.scale)
        resample_mode = Image.NEAREST if is_mask else Image.BICUBIC
        pil_img = pil_img.resize((newW, newH), resample=resample_mode)
        img = np.array(pil_img)
        if is_mask:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = mask_to_class(img, self.color_to_index)
            elif img.ndim == 2:
                img = img.astype(np.int64)
            return img
        else:
            # Ensure the image has 4 channels (RGB+NIR)
            if img.ndim == 2:
                # Single channel image: replicate it 4 times.
                img = np.stack([img] * 4, axis=-1)
            elif img.ndim == 3:
                if img.shape[2] == 3:
                    # 3-channel RGB: add a dummy NIR channel (zeros)
                    nir = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
                    img = np.concatenate([img, np.expand_dims(nir, axis=-1)], axis=-1)
                elif img.shape[2] != 4:
                    raise ValueError(f"Expected image with 1, 3, or 4 channels but got {img.shape[2]}")
            # Convert (H, W, C) to (C, H, W)
            img = img.transpose((2, 0, 1))
            if img.max() > 1:
                img = img / 255.0
            return img

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        mask_path = self.labels_list[idx]

        img = load_image(img_path)
        mask = load_image(mask_path)

        assert img.size == mask.size, f"Image and mask {img_path.stem} must have the same size"

        img = self.preprocess_image(img, is_mask=False)
        mask = self.preprocess_image(mask, is_mask=True)

        # Assert the image has 4 channels
        assert img.shape[0] == 4, f"Expected image with 4 channels, but got {img.shape[0]} channels for {img_path}"
        
        # Debug prints (you may want to remove these later)
        if idx == 0:
            print(f"Sample image shape: {img.shape}")
            print(f"Sample mask shape: {mask.shape}")

        return {
            'image': torch.as_tensor(img.copy()).float(),
            'mask': torch.as_tensor(mask.copy()).long()
        }
