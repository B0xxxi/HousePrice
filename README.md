# House Price Estimation

Нейросеть на PyTorch для оценки стоимости частного дома по фотографии фасада.

## Описание
Проект использует предобученную модель ResNet18 (Transfer Learning) для решения задачи регрессии. Модель обучается на датасете "Houses Dataset" (Ahmed and Moustafa).

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <your-repo-url>
   cd Neuro2
   ```

2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Использование

### 1. Загрузка данных
Скрипт автоматически скачает датасет и подготовит CSV файл:
```bash
python scripts/download_data.py
```

### 2. Обучение модели
Запустите обучение (по умолчанию 20 эпох):
```bash
PYTHONPATH=. python src/train.py
```
Лучшая модель будет сохранена в файл `best_model.pth`.

### 3. Предсказание
Для оценки стоимости дома используйте скрипт `predict.py`, передав путь к изображению:
```bash
PYTHONPATH=. python src/predict.py path/to/image.jpg
```

Пример:
```bash
PYTHONPATH=. python src/predict.py "data/Houses-dataset/Houses Dataset/100_frontal.jpg"
```

## Структура проекта
- `data/`: Данные (игнорируется git)
- `scripts/`: Вспомогательные скрипты (загрузка данных)
- `src/`: Исходный код (dataset, model, train, predict)
- `requirements.txt`: Зависимости
