import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)
        self._norm_fact = 1 / sqrt(output_size)

    def forward(self, joint):
        attn = torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2))*self._norm_fact  # (N, M, M)
        attn = F.softmax(attn, dim=-1)  # (N, M, M)

        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.W_h = nn.Linear(input_size, hidden_size)
        self.b_h = nn.Linear(hidden_size, hidden_size)
        self.u_h = nn.Parameter(torch.randn(hidden_size))  # learnable parameter u_h
        self.frelu = nn.functional.relu
        self.attn = SelfAttn(input_size,output_size)

    def forward(self, h):
        # 计算 e_i = tanh(W_h * h_i + b_h)
        # e = torch.tanh(self.W_h(h) + self.b_h(h))
        #
        # # 计算 a_i = exp(e_i^T * u_h) / sum(exp(e_i^T * u_h))
        # a = torch.exp(torch.matmul(e, self.u_h)) / torch.sum(torch.exp(torch.matmul(e, self.u_h)))
        #
        # # 计算 V = fRelu(sum(a_i * h_i))
        # V = self.frelu(torch.sum(a.unsqueeze(-1) * h, dim=1))
        a = self.attn(h)
        V = torch.sum(a, dim=1)
        return V

if __name__ == '__main__':

    # 使用模型
    batchsize = 10
    length = 5
    input_size = 5 # 输入维度
    hidden_size = 5 # 隐藏层维度
    output_size =  5# 输出维度

    # 创建模型实例
    model = MyModel(input_size, hidden_size, output_size)

    # 输入数据 h_i（例如，随机生成大小为 (num_samples, input_size) 的张量）
    h = torch.randn(batchsize, length,input_size)

    # 计算输出
    output = model(h)

    print("输出 V:", output)