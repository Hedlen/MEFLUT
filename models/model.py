import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from gfu.guided_filter import FastGuidedFilter, GuidedFilter
from models.model_base import CA, SA
import time
EPS = 1e-8

def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)

class Fusion(nn.Module):
    # end-to-end mef model
    def __init__(self, radius=7, eps=1, is_guided=True):
        super(Fusion, self).__init__()
        self.is_guided = is_guided
        if is_guided:
            self.gf = FastGuidedFilter(radius, eps)
    def forward(self, x_lr, w_lr, x_hr):
        if self.is_guided:
            w_hr = self.gf(x_lr, w_lr, x_hr)
        else:
            w_hr = F.upsample(w_lr, x_hr.size()[2:], mode='bilinear')

        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + EPS) / torch.sum((w_hr + EPS), dim=0)
        o_hr = torch.sum(w_hr * x_hr, dim=0, keepdim=True).clamp(0, 1)
        return o_hr, w_hr

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

class CFCA(nn.Module):
    def __init__(self, layers = 4, width = 64, reduction=8, n_frames = 3):
        super(CFCA, self).__init__()
        self.layer = layers
        self.width = width
        self.n_frames = n_frames
        self.reduction = reduction
        self.norm = AdaptiveNorm(self.width)
        self.head = nn.Sequential(
            nn.Conv2d(1, self.width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            self.norm,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            self.norm,
        )
        self.flca = CA(self.width, reduction=self.reduction)
        self.frw = CA(self.n_frames, reduction=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        x = self.head(x)
        x1 = self.flca(x)
        x = self.relu(x + x1)
        x = self.frw(x.transpose(0, 1)).transpose(0, 1)
        return x

class DISA(nn.Module):
    def __init__(self, layers = 4, width = 64):
        super(DISA, self).__init__()
        self.layer = layers
        self.width = width
        self.norm = AdaptiveNorm(width)
        self.head = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            self.norm,
            nn.LeakyReLU(0.2, inplace=True))

        self.body1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            self.norm,
            SA())
        self.body2 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=4,
                      dilation=4, bias=False),
            self.norm,
            SA())

        self.body3 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=8,
                      dilation=8, bias=False),
            self.norm,
            SA())

        self.tail = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=False),
            self.norm,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.width, 1, kernel_size=1, stride=1, padding=0, dilation=1)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv11 = nn.Conv2d(self.width * self.layer, self.width, kernel_size=1, stride=1, padding=0, dilation=1)


    def forward(self, x):
        x = self.head(x)
        x1 = self.body1(x)
        x2 = self.body2(x)
        x3 = self.body3(x)
        x5 = torch.cat([x1, x2, x3], 1)
        x6 = self.conv11(x5)
        x = self.relu(x6 + x)
        x = self.tail(x)
        return x

class MEFNetwork(nn.Module):
    def __init__(self, n_frames=3, radius=2, eps=1, is_guided=True, reduction=8, layers=2, width=48):
        super(MEFNetwork, self).__init__()
        self.n_frames = n_frames
        self.skeleton = CFCA(reduction=reduction, n_frames=self.n_frames, layers = layers, width=width)
        branch = [DISA(layers=layers, width=width) for _ in range(self.n_frames)]
        self.branch = nn.Sequential(*branch)
        self.is_guided = is_guided
        if is_guided:
            self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        x_lr_t = self.skeleton(x_lr)
        w = []
        for i in range(self.n_frames):
            w.append(self.branch[i](x_lr_t[i:i+1, :, :, :]))
        w_lr = torch.cat(w, 0)
        if self.is_guided:
            w_hr = self.gf(x_lr, w_lr, x_hr)
        else:
            w_hr = F.upsample(w_lr, x_hr.size()[2:], mode='bilinear')
        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + EPS) / torch.sum((w_hr + EPS), dim=0)
        o_hr = torch.sum(w_hr * x_hr, dim=0, keepdim=True).clamp(0, 1)
        return o_hr, w_hr

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))
