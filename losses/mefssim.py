import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / (gauss.sum())


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, window_size/6.).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(1, channel, window_size, window_size).contiguous()) / channel
    return window


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2).view(K, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2).view(K, H, W) \
                        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2).view(H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2).view(H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2).view(K, H, W) \
                - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q


def mef_ssim(X, Ys, window_size=11, is_lum=False):
    (_, channel, _, _) = Ys.size()
    window = create_window(window_size, channel)

    if Ys.is_cuda:
        window = window.cuda(Ys.get_device())
    window = window.type_as(Ys)

    return _mef_ssim(X, Ys, window, window_size, 0.08, 0.08, 0.01**2, 0.03**2, is_lum)


def mef_msssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False):
    # beta = torch.Tensor([0.0710, 0.4530, 0.4760])
    # beta = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    # beta = torch.Tensor([1, 1, 1, 1, 1])
    beta = torch.Tensor([1])
    if Ys.is_cuda:
        window = window.cuda(Ys.get_device())
        beta = beta.cuda(Ys.get_device())

    window = window.type_as(Ys)

    levels = beta.size()[0]
    l_i = []
    cs_i = []
    for _ in range(levels):
        l, cs = _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=is_lum, full=True)
        l_i.append(l)
        cs_i.append(cs)

        X = F.avg_pool2d(X, (2, 2))
        Ys = F.avg_pool2d(Ys, (2, 2))

    Ql = torch.stack(l_i)
    Qcs = torch.stack(cs_i)

    return (Ql[levels-1] ** beta[levels-1]) * torch.prod(Qcs ** beta)


class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


class MEF_MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEF_MSSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return mef_msssim(X, Ys, window, self.window_size,
                          self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)