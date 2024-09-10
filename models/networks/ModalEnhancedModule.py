import torch
import torch.nn as nn
from models.networks.Transformer.transformer import TransformerEncoder
import torch.nn.functional as F

class MEM(nn.Module):
    def __init__(self, input_dim, heads_num,layers_num):
        '''
            模态增强模块
        '''
        super(MEM, self).__init__()
        self.get_attetion = TransformerEncoder(input_dim, heads_num,layers_num)
        self.x_affline = nn.Linear(input_dim, input_dim)
        self.text_affline = nn.Linear(input_dim, input_dim)
        self.Manual_affline = nn.Linear(input_dim, input_dim)
        self.Trans_affline = nn.Linear(input_dim, input_dim)


        # 权重矩阵
        self.Wu = nn.Parameter(torch.randn(input_dim, input_dim))
        self.Wm = nn.Parameter(torch.randn(input_dim, input_dim))
        self.Wr = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x,source):
        '''
        x:增强目标特征，[length,batchsize,dim]
        source:增强源模态特征，[length,batchsize,dim]
        '''
        ## 手动增强
        text_future_attn = self.SigmoidAttn(source.transpose(0,1), self.Wr, self.Wm, self.Wu) #[batchsize,dim]
        Manual_x = torch.add(self.x_affline(x),self.text_affline(text_future_attn))
        Manual_x = F.softmax(torch.tanh(Manual_x),dim=-1)

        ## 自动增强
        Trans_x = self.get_attetion(x,source,source)
        # Dynamic filter
        weight_G = torch.sigmoid(self.Manual_affline(Manual_x) + self.Trans_affline(Trans_x))
        combined_x = Manual_x * weight_G + Trans_x * (1 - weight_G)
        return combined_x

    def SigmoidAttn(self,inputs,Wr, Wm, Wu):
        H_hot = torch.mul(torch.matmul(inputs.reshape(-1, inputs.shape[-1]), Wu),
                          torch.sigmoid(torch.matmul(inputs.reshape(-1, inputs.shape[-1]), Wm))) #[batch*length,dim]
        Att_a = F.softmax(torch.matmul(H_hot, Wr).reshape(inputs.shape[0], 1, inputs.shape[1]), dim=2)#[batch,1,length]
        temp_new = torch.matmul(Att_a, inputs)#[batch,1,dim]
        temp_new = temp_new.squeeze(1)#[batch,dim]
        return temp_new

# 第三版模态增强操作
class MEM2(nn.Module):
    def __init__(self, input_dim, heads_num,layers_num):
        '''
            模态增强模块
        '''
        super(MEM2, self).__init__()
        self.get_attetion = TransformerEncoder(input_dim, heads_num,layers_num)
        self.x_affline = nn.Linear(input_dim, input_dim,bias=False)
        self.text_affline = nn.Linear(input_dim, input_dim,bias=False)
        self.Manual_affline = nn.Linear(input_dim, input_dim)
        self.Trans_affline = nn.Linear(input_dim, input_dim,bias=False)


        # 权重矩阵
        self.Wu = nn.Parameter(torch.randn(input_dim, input_dim))
        self.Wm = nn.Parameter(torch.randn(input_dim, input_dim))
        self.Wr = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x,source):
        '''
        x:增强目标特征，[length,batchsize,dim]
        source:增强源模态特征，[length,batchsize,dim]
        '''
        ## 手动增强
        text_future_attn = self.SigmoidAttn(source.transpose(0,1), self.Wr, self.Wm, self.Wu) #[batchsize,dim]
        Manual_x = torch.add(self.x_affline(x),self.text_affline(text_future_attn))
        Manual_x = torch.tanh(Manual_x)

        ## 自动增强
        Trans_x = self.get_attetion(x,source,source)
        # Dynamic filter
        weight_G = torch.sigmoid(self.Manual_affline(Manual_x) + self.Trans_affline(Trans_x))
        combined_x = Manual_x * weight_G + Trans_x * (1 - weight_G)
        return combined_x

    def SigmoidAttn(self,inputs,Wr, Wm, Wu):
        H_hot = torch.mul(torch.matmul(inputs.reshape(-1, inputs.shape[-1]), Wu),
                          torch.sigmoid(torch.matmul(inputs.reshape(-1, inputs.shape[-1]), Wm))) #[batch*length,dim]
        Att_a = F.softmax(torch.matmul(H_hot, Wr).reshape(inputs.shape[0], 1, inputs.shape[1]), dim=2)#[batch,1,length]
        temp_new = torch.matmul(Att_a, inputs)#[batch,1,dim]
        temp_new = temp_new.squeeze(1)#[batch,dim]
        return temp_new
if __name__ == '__main__':
    x1 = torch.zeros(100,16,40)
    x2 = torch.randn_like(x1)
    MyTEM = MEM(40,1,1)
    x3 =MyTEM(x1,x2)
    print(x1)
    print(x2)
    print(x3.shape)
    print(x3)