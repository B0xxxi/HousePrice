import torch
from torchvision import transforms
from PIL import Image
from src.model import HousePriceModel
import sys
import os

def predict(image_path, model_path='best_model.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = HousePriceModel(pretrained=False) # No need to download weights again, we load ours
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
        
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        price = output.item()
        
    return price

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)
        
    price = predict(img_path)
    print(f"Predicted Price: ${price:,.2f}")
