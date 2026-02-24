#here goes dataset
import torch
from torchvision import datasets, transforms
import os

def get_dataloader(data_dir, batch_size, train=True):
    """
    Функция для создания загрузчика данных.
    data_dir: путь к папке, куда DVC положил данные (data/raw)
    """
    
    # 1. Определяем трансформации
    # ToTensor — превращает картинку (0-255) в тензор (0.0-1.0)
    # Normalize — вычитает среднее и делит на стандартное отклонение (ускоряет обучение)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Инициализируем датасет
    # download=False — КРИТИЧНО! Мы не хотим, чтобы скрипт качал данные сам.
    # Мы берем только то, что притащил нам DVC.
    dataset = datasets.MNIST(
        root=data_dir, 
        train=train, 
        download=False, 
        transform=transform
    )

    # 3. Создаем DataLoader
    # Он будет выдавать данные порциями (батчами)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train  # Перемешиваем только при обучении
    )
    
    return dataloader

if __name__ == "__main__":
    # Код для проверки (запустится только если запустить файл напрямую)
    # Проверим, видит ли скрипт данные после dvc pull
    try:
        test_loader = get_dataloader(data_dir='data/raw', batch_size=4, train=True)
        images, labels = next(iter(test_loader))
        print(f"Успех! Считан батч картинок размером: {images.shape}")
        print(f"Метки (цифры) в батче: {labels}")
    except Exception as e:
        print(f"Ошибка: Данные не найдены. Ты сделал dvc pull? \nДетали: {e}")