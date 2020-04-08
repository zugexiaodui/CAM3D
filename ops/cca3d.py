import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import CCA3D
import sys
sys.path.append('.')

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class _CCA_Weight3D(Function):
    @staticmethod
    def forward(ctx, t, f):
        N, C, T, H, W = t.size()
        weight = torch.ones([N, H + W + T - 2, T, H, W], dtype=t.dtype, layout=t.layout, device=t.device) # cuda
        CCA3D.ca_weight_forward(t, f, weight)
        ctx.save_for_backward(t, f)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        CCA3D.ca_weight_backward(dw.contiguous(), t, f, dt, df)
        _check_contiguous(dt, df)
        return dt, df


class _CCA_Map3D(Function):
    @staticmethod
    def forward(ctx, weight, g):
        out = torch.zeros_like(g)
        CCA3D.ca_map_forward(weight, g, out)
        ctx.save_for_backward(weight, g)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)

        CCA3D.ca_map_backward(dout.contiguous(), weight, g, dw, dg)

        _check_contiguous(dw, dg)

        return dw, dg

ca_weight3d = _CCA_Weight3D.apply
ca_map3d = _CCA_Map3D.apply

class _CrissCrossAttention3D(nn.Module):
    def __init__(self, in_dim):
        super(_CrissCrossAttention3D, self).__init__()
        inter_dim = in_dim // 4 # TODO!!!!
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight3d(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map3d(attention, proj_value)
        out = self.gamma * out #+ x
        return out

class RCCA3D_MODULE(nn.Module):
    def __init__(self, in_channels,recurrence=3):
        super(RCCA3D_MODULE, self).__init__()
        self.recurrence = recurrence
        inter_channels = in_channels# // 8
        print('RCCA3D: R=',self.recurrence)
        #self.conva = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 1, padding=0),
        #                           nn.BatchNorm3d(inter_channels))
        self.cca = _CrissCrossAttention3D(inter_channels)
        #self.convb = nn.Sequential(nn.Conv3d(inter_channels, in_channels, 1, padding=0),
        #                           nn.BatchNorm3d(in_channels))

    def forward(self, x):
        output = x
        for i in range(self.recurrence):
            output = self.cca(output)+x
        return output#+x


class CCA3DWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(CCA3DWrapper, self).__init__()
        self.block = block
        self.cca = RCCA3D_MODULE(block.bn3.num_features,recurrence=3)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.cca(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x

def make_cca3d(net,n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet) or isinstance(net, archs.small_resnet.ResNet):
        net.layer2 = nn.Sequential(
            net.layer2[0],
            net.layer2[1],
            CCA3DWrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
    else:
        raise NotImplementedError
