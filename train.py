import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from data import get_loaders
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as transforms
import os

# 配置
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
IMAGE_SIZE = 256
TRAIN_IMG_DIR = "dataset/sealand/train/images"
TRAIN_MASK_DIR = "dataset/sealand/train/masks"
VAL_IMG_DIR = "dataset/sealand/val/images"
VAL_MASK_DIR = "dataset/sealand/val/masks"
MODEL_SAVE_DIR = "saved_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据增强（训练和验证共用）
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def save_checkpoint(model, epoch, optimizer, loss):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }, f"{MODEL_SAVE_DIR}/unet_epoch{epoch}.pth")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def main():
    model = UNet(n_channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler()

    # 修复点：现在只传6个参数
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, transform
    )

    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # 验证（简化版）
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data, targets in val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                val_loss += loss_fn(model(data), targets).item()
            val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, epoch, optimizer, val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()