import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HousingPriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousingPriceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def main():
    # Загрузка данных
    print("Загрузка данных California Housing...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Размер датасета: {X.shape[0]} примеров, {X.shape[1]} признаков")
    print(f"Признаки: {housing.feature_names}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Создание DataLoader
    train_dataset = HousingDataset(X_train, y_train)
    test_dataset = HousingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Модель
    model = HousingPriceModel(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    print("\nНачало обучения...")
    epochs = 100
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # Сохраняем модель и scaler
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'feature_names': housing.feature_names
            }, 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    print(f"\nОбучение завершено! Лучший Test Loss: {best_test_loss:.4f}")
    print(f"Модель сохранена в 'best_model.pth'")
    
    # Тестовое предсказание
    model.eval()
    with torch.no_grad():
        sample = torch.FloatTensor(X_test[:5]).to(device)
        predictions = model(sample).cpu().numpy()
        actual = y_test[:5]
        
        print("\nПримеры предсказаний:")
        for i in range(5):
            print(f"Предсказано: ${predictions[i][0]*100000:.2f}, Реально: ${actual[i]*100000:.2f}")


if __name__ == "__main__":
    main()
