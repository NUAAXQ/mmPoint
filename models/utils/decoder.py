import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.utils.utils import MLP_Res, MLP_CONV

cudnn.benchnark=True
from models.utils.modules import *
from torch.nn import Conv1d

neg = 0.01
neg_2 = 0.2

class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = input
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x

class Edge_pointnet(nn.Module):
    def __init__(self, nk):
        super(Edge_pointnet, self).__init__()
        self.nk = nk

        dim = [3, 32, 64, 128]
        self.EdgeConv1 = EdgeBlock(dim[0], dim[1], self.nk)
        self.EdgeConv2 = EdgeBlock(dim[1], dim[2], self.nk)
        self.EdgeConv3 = EdgeBlock(dim[2], dim[3], self.nk)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)
        self.lrelu3 = nn.LeakyReLU(neg_2)

    def forward(self, x):

        x1 = self.EdgeConv1(x)
        x1 = self.lrelu1(x1)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)

        x3 = self.EdgeConv3(x2)
        x3 = self.lrelu3(x3)

        return x3

class Lift(nn.Module):
    def __init__(self, args, dim_feat=128, global_feat=True, i=0, radius = 1, up_factor=2, bounding=True):
        super(Lift, self).__init__()
        self.args = args
        self.nk = args.nk//2
        self.nz = args.nz
        self.global_feat = global_feat
        self.ps_dim = 32 if global_feat else 64
        self.bounding = bounding
        self.radius = radius
        self.i = i

        self.mlp_1 = Edge_pointnet(self.nk)
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat if self.global_feat else 128, layer_dims=[256, 128])
        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, self.ps_dim])
        self.ps = nn.ConvTranspose1d(self.ps_dim, 128, up_factor, up_factor, bias=False)
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_mmWave=None):
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)

        feat_1 = torch.cat([feat_1, torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),feat_mmWave.repeat(1, 1, feat_1.size(2))], 1)
        feat_2 = self.mlp_2(feat_1)

        feat_child = self.mlp_ps(feat_2)
        feat_child = self.ps(feat_child)

        feat_2_up = self.up_sampler(feat_2)

        K_curr = self.mlp_delta_feature(torch.cat([feat_child, feat_2_up], 1))
        delta = self.mlp_delta(torch.relu(K_curr))

        if self.bounding:
            delta = torch.tanh(delta) / self.radius**self.i

        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child

class Deform(nn.Module):
    def __init__(self, args):
        super(Deform, self).__init__()
        self.args = args
        self.nk = args.nk//2
        self.nz = args.nz
        self.neg = 0.01


        self.head = nn.Sequential(
            nn.Conv1d(3 + self.nz, 128, 1),
            nn.LeakyReLU(self.neg, inplace=True),
            nn.Conv1d(128, 128, 1),
            nn.LeakyReLU(self.neg, inplace=True),
        )

        self.mlp_1 = Edge_pointnet(self.nk)
        self.adain = AdaptivePointNorm(128, 128)

        self.tail = nn.Sequential(
            Conv1d(128, 64, 1),
            nn.LeakyReLU(self.neg, inplace=True),
            Conv1d(64, 32, 1),
            nn.LeakyReLU(self.neg, inplace=True),
            Conv1d(32, 3, 1),
            nn.Tanh()
        )

    def forward(self, pcd_lift, feat_mmWave=None):
        b, _, n_prev = pcd_lift.shape 

        feat = feat_mmWave.repeat(1, 1, n_prev)
        style = torch.cat([pcd_lift, feat], dim=1)
        style = self.head(style)

        feat_1 = self.mlp_1(pcd_lift)
        feat_deform = self.adain(feat_1, style)
        pcd_deform_delta = self.tail(feat_deform)

        pcd_deform = pcd_deform_delta + pcd_lift

        return pcd_deform