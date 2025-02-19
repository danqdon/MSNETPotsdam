# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from msnet import MSNet
from dataset import PotsdamSplitDataset
import os

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        assert images.shape[1] == 4, f"Batch images have {images.shape[1]} channels, expected 4."
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rutas de los CSV para el split de entrenamiento (ajusta según tu estructura)
    images_csv = "/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/train/images.csv"
    labels_csv = "/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/train/labels.csv"
    classes_json = "/mnt/e/ISPRS-Potsdam-adri/postdam_classes.json"
    # Si las rutas en el CSV son relativas, puedes definir el directorio base:
    base_dir = None

    dataset = PotsdamSplitDataset(images_csv, labels_csv, classes_json, scale=1.0, base_dir=base_dir)
    
    # División del dataset en entrenamiento (80%) y validación (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Número de clases: se obtiene del mapeo (la cantidad de colores definidos en el JSON)
    num_classes = len(dataset.color_to_index)
    model = MSNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 50
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Guardar checkpoint del modelo
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()
