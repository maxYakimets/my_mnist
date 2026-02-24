import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Первый сверточный слой: 1 входной канал (ЧБ), 32 фильтра, ядро 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        # Второй сверточный слой: 32 входа, 64 фильтра, ядро 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Dropout, чтобы нейронка не "зубрила" картинки (профилактика переобучения)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Полносвязные слои (классический классификатор в конце)
        # Число 9216 берется из размера картинки после сверток (12x12x64)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) # 10 выходов — для цифр от 0 до 9

    def forward(self, x):
        # Прогоняем данные через слои
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # Уменьшаем размер в 2 раза
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # "Выпрямляем" многомерную матрицу в один длинный вектор
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Softmax на выходе дает нам вероятности для каждого класса
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == "__main__":
    # Маленький тест: создаем модель и прогоняем через нее "пустую" картинку
    model = SimpleCNN()
    mock_data = torch.randn(1, 1, 28, 28)
    output = model(mock_data)
    print(f"Тест пройден! Размер выходного тензора: {output.shape}")