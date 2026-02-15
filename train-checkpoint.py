
# ===============================
# Reproducibility Settings
# ===============================
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ===============================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import optim,nn
from sklearn.model_selection import train_test_split

from torchmetrics import Accuracy

import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('train_test_data.csv')
df.head()

df.info()

df.describe()

df.isnull().sum()

df.columns

df.size

df.shape

x1 = df.iloc[: , 4:]
y1 = df.iloc[: , :4]

y1

x1

x = x1.values
y = y1.values

x

y

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 32)

mu = torch.mean(x_train , dim = 0)
std = torch.std(x_train , dim = 0)

mu

x_train = (x_train - mu) / std
x_test = (x_test - mu) / std

train_set = TensorDataset(x_train , y_train)
test_set = TensorDataset(x_test , y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False) 

input_size = x.shape[-1]  
output_size = y.shape[-1]  


model = nn.Sequential(
    # First layer
    nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

    # layer 1
    nn.Conv1d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Conv1d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm1d(64),

    # layer 2
    nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Conv1d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm1d(128),
    nn.Conv1d(128, 128, kernel_size=1, stride=2),  # Shortcut (corrected)

    # layer 3
    nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Conv1d(256, 256, kernel_size=3, padding=1),
    nn.BatchNorm1d(256),
    nn.Conv1d(256, 256, kernel_size=1, stride=2),  # Shortcut (corrected)

    # layer 4
    nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Conv1d(512, 512, kernel_size=3, padding=1),
    nn.BatchNorm1d(512),
    nn.Conv1d(512, 512, kernel_size=1, stride=2),  # Shortcut (corrected)

    # Final layer
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(512, 4)  # خروجی 4 (تعداد خروجی‌های پیش‌بینی شده)
)


loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Accuracy



# Matrix
loss_train_hist = []
loss_test_hist = []

acc_train_hist = []
acc_test_hist = []


num_epochs = 80



for num in range(num_epochs):
    model.train()  # مدل را در حالت آموزش قرار می‌دهیم
    train_loss = 0
    for x_batch , y_batch in train_loader:
        # ورودی‌ها به فرمت (batch_size, channels, length) هستند
        x_batch = x_batch.unsqueeze(1)  # افزودن بعد کانال (در اینجا 1 کانال)
        
        # انجام پیش‌بینی
        y_pre = model(x_batch)
        
        # محاسبه loss
        loss = loss_fn(y_pre, y_batch)
        
        # بهینه‌سازی
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        loss_train_hist.append(train_loss / len(train_loader))

    print(f"Epoch [{num+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader)}")

    # تست مدل
    model.eval()  # مدل را در حالت ارزیابی قرار می‌دهیم
    test_loss = 0
    with torch.no_grad():  # بدون محاسبه گرادیان
        for x_batch1, y_batch1 in test_loader:
            x_batch1 = x_batch1.unsqueeze(1)  # افزودن بعد کانال (در اینجا 1 کانال)

            # پیش‌بینی
            y_pre1 = model(x_batch1)

            # محاسبه loss
            loss = loss_fn(y_pre1, y_batch1)
            test_loss += loss.item()

            loss_test_hist.append(test_loss / len(test_loader))

        print(f"Epoch [{num+1}/{num_epochs}], Test Loss: {test_loss/len(test_loader)}")
        print(' ------------------------------------------------------------- ')

print(f"Test Loss: {test_loss/len(test_loader)}")

# z = x[1:2 , 4:]

df2 = pd.read_csv('n2.csv')

df2 = df2.values

df2 = torch.FloatTensor(df2)
df2.shape

y1.shape

new_data = torch.FloatTensor(df2) 
new_data = new_data.unsqueeze(1)  
model.eval()  
with torch.no_grad():  
    predictions = model(new_data)

print("Predictions: ", predictions)

