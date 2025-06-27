# agents/d3qn/model.py

import torch
import torch.nn as nn

class D3QN(nn.Module):
    """
    Dueling Double DQN 的网络结构。
    输入是 (N, C, H, W) 的图像，输出是每个动作的 Q 值。
    """
    def __init__(self, input_shape, num_actions):
        super().__init__()
        
        # CNN 特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算 CNN 输出的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).squeeze(-1)  
            cnn_output_dim = self.cnn(dummy_input).flatten().shape[0]

        # Dueling 架构
        # 1. 价值流 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # 输出 V(s)
        )
        
        # 2. 优势流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) # 输出 A(s, a)
        )

    def forward(self, x):
        if x.shape[-1] == 1 and x.ndim == 5:
            x = x.squeeze(-1)
        
        x = x / 255.0
        
        features = self.cnn(x)
        features = features.view(features.size(0), -1) # Flatten
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 组合 V(s) 和 A(s, a) 得到 Q(s, a)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values