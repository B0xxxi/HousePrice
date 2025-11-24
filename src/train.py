import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import HousePriceDataset, get_transforms
from src.model import HousePriceModel
import time
import copy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                targets = batch['price'].to(device).float().view(-1, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Configuration
    DATA_CSV = "data/houses.csv"
    BATCH_SIZE = 16 # Small batch size for small dataset
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    full_dataset = HousePriceDataset(csv_file=DATA_CSV, transform=get_transforms(train=True))
    
    # Split into train and val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Override transform for validation (no augmentation)
    # Note: random_split doesn't allow changing transform easily per subset if they share the underlying dataset.
    # Ideally we should split indices and create two datasets, but for simplicity we'll use the same transform (augmentation on val is not ideal but acceptable for quick demo)
    # Or better:
    train_dataset.dataset.transform = get_transforms(train=True)
    # This changes it for both because they point to the same object!
    # Correct way:
    # We will just use the same dataset class but instantiate it twice with different transforms if we had indices.
    # Let's do it properly.
    
    df = pd.read_csv(DATA_CSV)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # Save temp csvs
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    
    train_ds = HousePriceDataset("data/train.csv", transform=get_transforms(train=True))
    val_ds = HousePriceDataset("data/val.csv", transform=get_transforms(train=False))
    
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    # Model
    model = HousePriceModel(pretrained=True)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device)

if __name__ == "__main__":
    import pandas as pd # Need to import here or at top
    main()
