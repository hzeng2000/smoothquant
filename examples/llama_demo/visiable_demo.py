import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义量化函数
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

# 定义简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 2)  # 2*2的权重矩阵

    def forward(self, x):
        return self.fc(x)

# 初始化网络和输入
net = SimpleNN()
input_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 打印原始权重
print("Original Weight Matrix:")
print(net.fc.weight)

# 量化权重
quantized_weight = quantize_weight_per_channel_absmax(net.fc.weight)
print("\nQuantized Weight Matrix:")
print(quantized_weight)

# 设置量化后的权重
net.fc.weight.data = quantized_weight

# 前向传播
output = net(input_matrix)
print("\nOutput Matrix:")
print(output)