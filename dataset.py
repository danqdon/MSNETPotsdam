# dataset.py
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_image(filename):
    return Image.open(filename)

class PotsdamDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, classes_json: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale

        # Cargar la definición de clases desde el JSON
        with open(classes_json, 'r') as f:
            self.classes = json.load(f)
        # Asumimos que el JSON define los valores únicos de máscara (por ejemplo, [0, 1, 2, ...])
        self.mask_values = sorted(self.classes.values())
        
        # Lista de IDs (asumiendo que los nombres de archivo sin extensión coinciden entre imagen y máscara)
        self.ids = [f.stem for f in self.images_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No se encontraron imágenes en {images_dir}')

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.array(pil_img)
        if is_mask:
            # Suponiendo que la máscara es de 2 dimensiones
            mask = np.zeros((newH, newW), dtype=np.int64)
            # Se asigna un índice a cada valor encontrado en la máscara
            for i, v in enumerate(sorted(np.unique(img))):
                mask[img == v] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                # Para imágenes con 4 canales (RGB+NIR), se debe tener en cuenta el orden
                img = img.transpose((2, 0, 1))
            # Normalización simple (si los valores son 0-255)
            if img.max() > 1:
                img = img / 255.0
            return img

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        # Buscamos la imagen y la máscara según el nombre
        img_path = list(self.images_dir.glob(name + '.*'))[0]
        mask_path = list(self.mask_dir.glob(name + '.*'))[0]

        img = load_image(img_path)
        mask = load_image(mask_path)

        assert img.size == mask.size, f'La imagen y la máscara {name} deben tener el mismo tamaño'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float(),
            'mask': torch.as_tensor(mask.copy()).long()
        }
