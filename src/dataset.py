import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HousePriceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx]['image_path']
        price = self.data_frame.iloc[idx]['price']

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
             # Fallback for missing images if any slipped through
             # In a real scenario, we might want to log this or handle it better
             # For now, return a black image or skip (skipping in __getitem__ is hard)
             print(f"Warning: Could not open {img_path}")
             image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
        
        # Return price as float tensor
        # We might want to normalize price later, but for now raw price
        sample = {'image': image, 'price': torch.tensor(float(price), dtype=torch.float32)}

        return sample

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
