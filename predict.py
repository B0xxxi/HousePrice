import torch
import numpy as np
from housing_model import HousingPriceModel


def predict_price(features):
    """
    Предсказание цены жилья
    
    features: список или массив с признаками
    [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    """
    # Загрузка модели
    checkpoint = torch.load('best_model.pth', weights_only=False)
    scaler = checkpoint['scaler']
    feature_names = checkpoint['feature_names']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HousingPriceModel(input_size=len(feature_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Нормализация входных данных
    features_scaled = scaler.transform([features])
    
    # Предсказание
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        prediction = model(features_tensor).cpu().numpy()[0][0]
    
    return prediction * 100000  # Конвертация в доллары


if __name__ == "__main__":
    # Пример использования
    print("Признаки: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]")
    
    # Пример: средний дом в Калифорнии
    example_features = [3.5, 25, 5.5, 1.2, 1500, 3.0, 35.0, -119.0]
    
    price = predict_price(example_features)
    print(f"\nПредсказанная цена: ${price:.2f}")
