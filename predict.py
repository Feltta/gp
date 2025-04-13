import torch
from PIL import Image
from unet import UNet
import torchvision.transforms as transforms
import numpy as np

def predict():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "saved_models/unet_epoch48.pth"
    IMAGE_PATH = "dataset/sealand/val/images/000201.png"
    OUTPUT_PATH = "prediction.png"

    model = UNet(n_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        pred_prob = torch.sigmoid(logits)
        pred_mask = (pred_prob > 0.5).float()

    Image.fromarray((pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)).save(OUTPUT_PATH)
    print(f"预测结果已保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    predict()