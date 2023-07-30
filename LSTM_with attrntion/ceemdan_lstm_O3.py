# 1. 导入必要的库
import torch
import torch.nn as nn

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

         # Initialize the attention weights
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(x,(h0,c0))
        # Apply attention mechanism
        attn_weights = torch.matmul(lstm_out, self.w_omega)
        attn_weights = torch.tanh(attn_weights)
        attn_weights = torch.matmul(attn_weights, self.u_omega)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_out = lstm_out * attn_weights
        attn_out = attn_out.sum(dim=1)
        # Pass the attention output through the fully connected layer
        out = self.dropout(attn_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    input_size = 1
    hidden_size = 1
    num_layers = 1
    output_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers,output_size).to(device)
    X = torch.rand(size=(1, 4, 1), dtype=torch.float32)
    X = X.to(device)
    print(model(X))