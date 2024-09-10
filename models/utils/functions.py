from torch.autograd import Function
import torch.nn as nn
import torch

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)  # 按维度求均值，keepdims=True保持转换后维度不变
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()  # 求范数
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class KL(torch.nn.Module):
    def __init__(self, reduction):
        super(KL, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, var_1, var_2): # var is standard deviation
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = var_1.type(dtype=torch.float64)
        sigma_2 = var_2.type(dtype=torch.float64)

        sigma_diag_1 = torch.diag_embed(sigma_1, offset=0, dim1=-2, dim2=-1)
        sigma_diag_2 = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1)

        sigma_diag_2_inv = sigma_diag_2.inverse()

        # log(det(sigma2^T)/det(sigma1))
        term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()
        # term_1[term_1.ne(term_1)] = 0

        # trace(inv(sigma2)*sigma1)
        term_2 = torch.diagonal((torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3 = torch.matmul(torch.matmul((mu2 - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_2_inv),
                              (mu2 - mu1).unsqueeze(-1)).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n = mu1.shape[1]

        # Calc kl divergence on entire batch
        kl = 0.5 * (term_1 - n + term_2 + term_3)

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')

        return kl_agg

# class KL(torch.nn.Module):
#     def __init__(self):
#         super(KL, self).__init__()
#         self.loss_func = MultVariateKLD(reduction='mean')
#
#     def foward(self,A_F,V_F,L_F):
#         var_a, var_v, var_t = A_F.var(dim=-1), V_F.var(dim=-1), L_F.var(dim=-1)
#         a_, v_, l_ = A_F.mean(-1), V_F.mean(-1), L_F.mean(-1)
#
#         output_av = (self.loss_func(a_, v_, var_a, var_v) + self.loss_func(v_, a_, var_v, var_a)) / 2
#         output_at = (self.loss_func(a_, l_, var_a, var_t) + self.loss_func(l_, a_, var_t, var_a)) / 2
#         output_vt = (self.loss_func(v_, l_, var_v, var_t) + self.loss_func(l_, v_, var_t, var_v)) / 2
#
#         loss_all = (output_at + output_av + output_vt) / 3
#
#         return loss_all