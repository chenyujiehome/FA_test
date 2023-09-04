import torch
import torch.nn as nn
import time

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# 随机生成输入数据
input_data = torch.randn(64, 1000).cuda()
target_data = torch.randn(64, 1000).cuda()

# 测量 forward pass 时间
start_time = time.time()
outputs = model(input_data)
end_time = time.time()
print(f"Forward pass time: {end_time - start_time} seconds")

# 测量 backward pass 时间
loss_function = nn.MSELoss()
loss = loss_function(outputs, target_data)
start_time = time.time()
loss.backward()
optimizer.step()
end_time = time.time()
print(f"Backward pass time: {end_time - start_time} seconds")
