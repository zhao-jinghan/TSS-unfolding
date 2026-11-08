import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class DropPath(nn.Module):
    """DropPath aka Stochastic Depth."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast (B,1,1,..)  ndim=x.ndim
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # torch.rand生成[0,1)中均匀分布的随机数
        random_tensor.floor_()  # binary mask
        output = x.div(keep_prob) * random_tensor
        return output

class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, drop_path_rate=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, src, *args, **kwargs):
        x = src
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.drop_path(self.dropout1(x2))  # Apply DropPath
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.drop_path(self.dropout2(x2))  # Apply DropPath
        return self.norm2(x)
