import torch
import torch.nn.functional as F
import yaml
import os
from model import SimpleCNN
from dataset import get_dataloader

def train():
    # 1. Загружаем параметры из нашего YAML-конфига
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Извлекаем секцию 'train' (как мы договорились в нотации YAML)
    params = config["train"]
    
    # 2. Выбираем устройство: видеокарта (cuda) или процессор (cpu)
    # В Colab тут будет cuda, в WSL — скорее всего cpu (если нет NVIDIA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Обучение на устройстве: {device} ---")

    # 3. Готовим данные
    # Путь 'data/raw' — это то, что контролирует DVC
    train_loader = get_dataloader(
        data_dir='data/raw', 
        batch_size=params["batch_size"], 
        train=True
    )

    # 4. Инициализируем нейронку и отправляем её на устройство
    model = SimpleCNN().to(device)
    
    # Выбираем оптимизатор (Adam — отличный стандарт)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # 5. Главный цикл обучения
    model.train()
    print(f"Начинаем обучение: {params['epochs']} эпох...")

    for epoch in range(1, params["epochs"] + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Переносим порцию данных на видеокарту/процессор
            data, target = data.to(device), target.to(device)
            
            # Обнуляем градиенты (чтобы ошибки не накапливались с прошлого шага)
            optimizer.zero_grad()
            
            # Прямой ход (Forward Pass)
            output = model(data)
            
            # Считаем ошибку (Loss)
            loss = F.nll_loss(output, target)
            
            # Обратный ход (Backpropagation) — считаем, куда крутить веса
            loss.backward()
            
            # Шаг обновления весов
            optimizer.step()

            # Печатаем прогресс каждые 100 батчей
            if batch_idx % 100 == 0:
                print(f"Эпоха {epoch} | Батч {batch_idx} | Loss: {loss.item():.4f}")

    # 6. Сохранение результата
    # Создаем папку models, если её еще нет
    os.makedirs("models", exist_ok=True)
    
    # Сохраняем только веса модели (state_dict) — это стандарт индустрии
    torch.save(model.state_dict(), "models/model.pth")
    print(f"--- Успех! Модель сохранена в models/model.pth ---")

if __name__ == "__main__":
    train()