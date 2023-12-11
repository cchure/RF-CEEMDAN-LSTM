# 1. 导入必要的库
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ceemdan_lstm_O3 import LSTMModel
import torch.optim.lr_scheduler as lr_scheduler

# 加载数据集并进行必要的数据预处理
df = pd.read_csv(r'C:\python training\小论文相关\CEEMDAN\O3\ceemdan_lstm_O3.csv')
df.dropna(inplace=True)  # 删除NaN值

# 设置计算数据的设备，有GPU用GPU，没有GPU用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 提取所有特征列和目标列
features = df.iloc[:, :-1].values
target = df.iloc[:, -1:].values

# 对特征和目标进行归一化处理
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
target = scaler.fit_transform(target)

# 3. 划分数据集为训练集和测试集
train_size = int(len(features) * 0.8)
test_size = len(features) - train_size
train_features = torch.tensor(features[:train_size, :])
train_target = torch.tensor(target[:train_size, :])
test_features = torch.tensor(features[train_size:, :])
test_target = torch.tensor(target[train_size:, :])
# 将数据集封装为PyTorch的Dataset和DataLoader
class MyDataset(Dataset):
    def __init__(self, features, target,lookback):
        self.features = features
        self.target = target
        self.lookback = lookback

    def __len__(self):
        return len(self.features)- self.lookback
    
    def __getitem__(self, index):
        return torch.as_tensor(self.features[index:index+self.lookback], dtype=torch.float32), torch.as_tensor(self.target[index+self.lookback], dtype=torch.float32)

lookback = 15
batch_size=16
train_dataset = MyDataset(train_features, train_target,lookback=lookback)
test_dataset = MyDataset(test_features, test_target,lookback=lookback)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


input_size=17
hidden_size = 128
output_size = 1
num_layers = 2

#  定义损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.005)

#  训练模型并进行测试
#  训练模型并进行测试
best_acc=0
epochs=500
for epoch in range(1, epochs+1):
    train_loss = 0
    model.train()
    for features, target in train_loader:
        features=features.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output = model(features.float())
        loss = criterion(output, target.float())

        loss.backward()
        optimizer.step()
        #scheduler.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    if epoch %10 == 0:
        print('Epoch {} | Train Loss: {:.6f}'.format(epoch, train_loss）
            

    # 保存最佳模型
    best_loss=1
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), r'C:\python training\小论文相关\CEEMDAN\O3\save_weights\best_model_01.pt')  
