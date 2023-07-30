import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ceemdan_lstm_O3 import LSTMModel
import matplotlib.pyplot as plt

# 2. 加载数据集并进行必要的数据预处理
df = pd.read_csv(r'C:\python training\小论文相关\CEEMDAN\O3\ceemdan_lstm_O3.csv')
df.dropna(inplace=True)  # 删除NaN值

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

train_dataset = MyDataset(train_features, train_target,lookback=lookback)
test_dataset = MyDataset(test_features, test_target,lookback=lookback)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


if __name__ == '__main__':
    # 选择计算的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_size=17
    hidden_size = 128
    output_size = 1
    num_layers = 2
    # 加载需要的模型
    model = LSTMModel(input_size, hidden_size, num_layers,output_size).to(device)
    model.load_state_dict(torch.load(r'C:\python training\小论文相关\CEEMDAN\O3\save_weights\best_model_01.pt'))

    # 设置归一化后保存结果的列表
    real = []
    prediction = []

model.eval()
test_loss = 0
predictions = []
with torch.no_grad():
    for features, target in test_loader:
        features=features.to(device)
        target=target.to(device)

        output = model(features)
        output=output.cpu().numpy()
        predictions.append(output)
    predictions = np.concatenate(predictions, axis=0)
    predictions = scaler.inverse_transform(predictions)
    test_target = scaler.inverse_transform(test_target)
    test_target =test_target[15:,:]


    # 调用模型评价指标
    # R2
    from sklearn.metrics import r2_score
    # MSE
    from sklearn.metrics import mean_squared_error
    # MAE
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_absolute_percentage_error

    # 计算模型的评价指标
    R2 = r2_score(test_target, predictions)
    MAE = mean_absolute_error(test_target, predictions)
    RMSE = np.sqrt(mean_squared_error(test_target, predictions))
    MAPE = mean_absolute_percentage_error(test_target, predictions)


    # 打印模型的评价指标
    print('R2:', R2)
    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)

 #导入到csv文件
df_pred = pd.DataFrame(predictions, columns=['Predicted O3'])
df_true = pd.DataFrame(test_target, columns=['True O3'])
df_result = pd.concat([df_true, df_pred], axis=1)
df_result.to_csv(r'C:\python training\小论文相关\CEEMDAN\O3\prediction_O3_1.csv', index=True)   
  