
# Создайте файл download_data.py
import torch
from torchvision import datasets

# Скачиваем в папку data/raw
train_set = datasets.MNIST('./data/raw', train=True, download=True)
test_set = datasets.MNIST('./data/raw', train=False, download=True)
print("Данные MNIST успешно скачаны!")