import torch
import torch.nn as nn

# 定义输入数据
batch_size = 16
in_channels = 3
input_length = 10
x = torch.randn(batch_size, in_channels, input_length)

# 定义一维卷积层
out_channels = 6
kernel_size = 3
conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)

# 执行卷积操作
output = conv1d(x)
print(output.shape)  # 输出形状: (batch_size, out_channels, output_length)