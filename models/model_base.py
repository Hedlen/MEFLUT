
import torch
import torch.nn as nn
import torch.nn.functional as F

##---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SA(nn.Module):
    def __init__(self, kernel_size=5):
        super(SA, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class CA(nn.Module):
    def __init__(self, channel, reduction=2, bias=False):
        super(CA, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class triple_bottleneck(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(triple_bottleneck, self).__init__()
        self.block1 = Bottleneck(inChannels, outChannels)
        self.block2 = Bottleneck(inChannels, outChannels)
        self.block3 = Bottleneck(inChannels, outChannels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class up(nn.Module):

    def __init__(self, inChannels, outChannels, scale_factor=2):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2*outChannels, outChannels, 3, stride=1, padding=1)
        self.scale_factor = scale_factor

    def forward(self, x, skipConnection):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

        diffY = torch.tensor([skipConnection.size()[2] - x.size()[2]])
        diffX = torch.tensor([skipConnection.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skipConnection), 1)), negative_slope=0.1)
        return x


class down(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernelSize, stride=1, padding=int((kernelSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernelSize, stride=1, padding=int((kernelSize - 1) / 2))

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x= F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class res_up(nn.Module):
    def __init__(self, inChannels, outChannels, scale_factor=2):
        super(res_up, self).__init__()
        self.triple_bottleneck = triple_bottleneck(inChannels, outChannels)
        self.conv = nn.Conv2d(inChannels, int(inChannels / 4), kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x, skipConnection):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

        diffY = torch.tensor([skipConnection.size()[2] - x.size()[2]])
        diffX = torch.tensor([skipConnection.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = self.triple_bottleneck(torch.cat((x, skipConnection), 1))
        x = self.conv(x)
        return x


class res_down(nn.Module):
    def __init__(self, inChannels, outChannels, change_channel=True):
        super(res_down, self).__init__()
        self.down = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=2, padding=1)
        self.triple_bottleneck = triple_bottleneck(inChannels, outChannels)
        self.conv = nn.Conv2d(inChannels, inChannels * 2, kernel_size=1)
        self.change_channel = change_channel

    def forward(self, x):
        x = self.down(x)
        x = self.triple_bottleneck(x)
        if self.change_channel:
            x = self.conv(x)
        return x

class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):

    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []

        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        # gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img


