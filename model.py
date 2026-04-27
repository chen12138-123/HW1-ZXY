import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims, num_classes, activation='relu', dropout_rate=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_size
        
        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError("Unsupported activation")

        # 构建隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # 添加 Batch Normalization
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate)) # 添加 Dropout
            prev_dim = h_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 将图片展平
        x = x.view(x.size(0), -1)
        return self.network(x)
