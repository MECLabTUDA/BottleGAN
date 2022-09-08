from .sb_gans import  AdaIN
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Unet Style

class DoubleConvDownInst(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if in_channels != out_channels:
            self.bottle = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.bottle = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DoubleConvDownInstAda(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, style_dim=64):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = AdaIN(style_dim=style_dim, num_features=mid_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaIN(style_dim=style_dim, num_features=out_channels)
        self.act2 = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.bottle = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.bottle = nn.Identity()

    def forward(self, x, s):
        x = self.conv1(x)
        x = self.norm1(x,s)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x,s)
        x = self.act2(x)
        return x

class DoubleConvUpInstAda(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, style_dim=64):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = AdaIN(style_dim=style_dim, num_features=mid_channels)
        self.act1 =  nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.bottle = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.bottle = nn.Identity()

    def forward(self, x, s):
        x = self.conv1(x)
        x = self.norm1(x,s)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class DoubleConvUpInst(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, style_dim=64):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(mid_channels)
        self.act1 =  nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.bottle = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.bottle = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class DownInst(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDownInst(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownInstAda(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, style_dim=64):
        super().__init__()
        self.avgpool_conv = nn.MaxPool2d(2)
        self.conv = DoubleConvDownInstAda(in_channels, out_channels, style_dim=style_dim)
        

    def forward(self, x, s):
        x = self.avgpool_conv(x)
        return self.conv(x, s)

class UpInstAda(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, style_dim=64):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConvUpInstAda(in_channels, out_channels, in_channels // 2, style_dim=style_dim)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvUpInstAda(in_channels, out_channels, style_dim=style_dim)

    def forward(self, x1, x2, s):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, s)

class UpInst(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, style_dim=64):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConvUpInst(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvUpInst(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConvInst(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim=128):
        super(OutConvInst, self).__init__()

        self.norm = AdaIN(num_features=in_channels, style_dim=style_dim)
        self.to_rgb  = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, s):
        x = self.norm(x,s)
        return self.to_rgb(x)

def convert_RGB_to_OD(I):
    return - torch.log(I + 1e-12)

def convert_OD_to_RGB(OD):
    return torch.exp(-OD - 1e-12) * 1

class UNetStyleDown(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_size=64, bilinear=True, linear=False):
        super(UNetStyleDown, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_down = 2
        self.linear = linear
        self.in_norm = nn.InstanceNorm2d(3)

        self.inc = DoubleConvDownInst(n_channels * 1, 64 // self.channel_down)
        self.down1 = DownInstAda(64// self.channel_down, 128 // self.channel_down, style_dim=embedding_size)
        self.down2 = DownInstAda(128 // self.channel_down, 256 // self.channel_down, style_dim=embedding_size)
        self.down3 = DownInstAda(256 // self.channel_down, 512 // self.channel_down, style_dim=embedding_size)
        factor = 2 if bilinear else 1
        self.down4 = DownInstAda(512 // self.channel_down, 1024 // factor // self.channel_down, style_dim=embedding_size)
        self.up1 = UpInstAda(1024 // self.channel_down, 512 // factor // self.channel_down, bilinear, style_dim=embedding_size)
        self.up2 = UpInstAda(512 // self.channel_down, 256 // factor // self.channel_down, bilinear, style_dim=embedding_size)
        self.up3 = UpInstAda(256 // self.channel_down, 128 // factor // self.channel_down, bilinear, style_dim=embedding_size)
        self.up4 = UpInstAda(128 // self.channel_down, 64, bilinear, style_dim=embedding_size)
        self.outc = OutConvInst(64, n_classes, style_dim=embedding_size)

    def forward(self, x, s, mt, c_ratio):
        if not self.linear:
            x = self.in_norm(x)
            x1 = self.inc(x)
            x2 = self.down1(x1,s)
            x3 = self.down2(x2,s)
            x4 = self.down3(x3,s)
            x5 = self.down4(x4,s)
            x = self.up1(x5, x4, s)
            x = self.up2(x, x3, s)
            x = self.up3(x, x2, s)
            x = self.up4(x, x1, s)
            x = torch.sigmoid(self.outc(x, s))
        else:
            x_shape = x.shape
            x = torch.bmm(x.permute(0,2,3,1).reshape(x_shape[0],-1, 3), mt)
            x = x.reshape(x_shape[0],x_shape[2], x_shape[3], 3).permute(0,3,1,2)
            x = torch.clip(x, 0, 1)

        return x

class UNetStyleUp(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_size=64, bilinear=True, linear=False):
        super(UNetStyleUp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_down = 2
        self.linear = linear
        self.in_norm = nn.InstanceNorm2d(3)
        
        self.inc = DoubleConvDownInst(n_channels * 1, 64 // self.channel_down)
        self.down1 = DownInstAda(64// self.channel_down, 128 // self.channel_down, style_dim=embedding_size)
        self.down2 = DownInstAda(128 // self.channel_down, 256 // self.channel_down, style_dim=embedding_size)
        self.down3 = DownInstAda(256 // self.channel_down, 512 // self.channel_down, style_dim=embedding_size)
        factor = 2 if bilinear else 1
        self.down4 = DownInstAda(512 // self.channel_down, 1024 // factor // self.channel_down, style_dim=embedding_size)
        self.up1 = UpInstAda(1024 // self.channel_down, 512 // factor // self.channel_down, bilinear, style_dim=embedding_size)
        self.up2 = UpInstAda(512 // self.channel_down, 256 // factor // self.channel_down, bilinear, style_dim=embedding_size)
        self.up3 = UpInstAda(256 // self.channel_down, 128 // factor // self.channel_down, bilinear, style_dim=embedding_size)
        self.up4 = UpInstAda(128 // self.channel_down, 64, bilinear, style_dim=embedding_size)
        self.outc = OutConvInst(64, n_classes, style_dim=embedding_size)

    def forward(self, x, s, ms, c_ratio):
        if not self.linear:
            x = self.in_norm(x)
            x1 = self.inc(x)
            x2 = self.down1(x1,s)
            x3 = self.down2(x2,s)
            x4 = self.down3(x3,s)
            x5 = self.down4(x4,s)
            x = self.up1(x5, x4, s)
            x = self.up2(x, x3, s)
            x = self.up3(x, x2, s)
            x = self.up4(x, x1, s)
            x = torch.sigmoid(self.outc(x,s))
        else:
            x_shape = x.shape
            x = torch.bmm(x.permute(0,2,3,1).reshape(x_shape[0],-1, 3), ms)
            x = x.reshape(x_shape[0],x_shape[2], x_shape[3], 3).permute(0,3,1,2)
            x = torch.clip(x, 0, 1)
        return x

class Conv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)

        self.w = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        nn.init.dirac_(self.w.weight.data)
        self.std = nn.Parameter(torch.zeros_like(self.weight.data), requires_grad=False)
        self.gauss_scale = 0.2


    def forward(self, input):
        self.weight.data += (torch.normal(0, torch.nan_to_num(self.std) + 1e-6) * self.gauss_scale)
        outputs = super().forward(self.w(input))
        return outputs

class SBLayer(nn.Module):
    def __init__(self, style_dim, num_features, out_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc1 = nn.Linear(style_dim, num_features*2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(num_features*2, num_features*2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(num_features*2, num_features*2)
        self.conv = nn.Conv2d(num_features, out_dim, kernel_size=1, padding=0)

    def forward(self, x, s):
        h = self.fc1(s)
        h = self.act1(h)
        h = self.fc2(h)
        h = self.act2(h)
        h = self.fc3(h)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        x = (1 + gamma) * self.norm(x) + beta
        x = self.conv(x)
        return torch.sigmoid(x)

class SBNet(nn.Module):
    def __init__(self, channels=3, embedding_size=64, linear=False, hidden_dim=256):
        super().__init__()
        feature_dims = [(channels, hidden_dim), (hidden_dim, hidden_dim), (hidden_dim, hidden_dim),  (hidden_dim, hidden_dim), (hidden_dim, channels)]
        self.linear = linear
        self.layers = nn.ModuleList([SBLayer(style_dim=embedding_size, num_features=feature_dims[i][0], out_dim=feature_dims[i][1]) for i in range(len(feature_dims))])


    def forward(self, x, s, ms, c_ratio):
        if not self.linear:
            x = convert_RGB_to_OD(x)
            for layer in self.layers:
                x = layer(x, s)

            #x =  x_c + x
            x = convert_OD_to_RGB(x)
        else:
            x_shape = x.shape
            x = torch.bmm(x.permute(0,2,3,1).reshape(x_shape[0],-1, 3), ms)
            x = x.reshape(x_shape[0],x_shape[2], x_shape[3], 3).permute(0,3,1,2)
            x = torch.clip(x, 0, 1)
        return x

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Unet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            Conv2D(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if in_channels != out_channels:
            self.bottle = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.bottle = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, samples=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_down = 2
        self.return_features = False
        self.samples = samples

        self.inc = DoubleConv(n_channels, 64 // self.channel_down)
        self.down1 = Down(64// self.channel_down, 128 // self.channel_down)
        self.down2 = Down(128 // self.channel_down, 256 // self.channel_down)
        self.down3 = Down(256 // self.channel_down, 512 // self.channel_down)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // self.channel_down, 1024 // factor // self.channel_down)
        self.up1 = Up(1024 // self.channel_down, 512 // factor // self.channel_down, bilinear)
        self.up2 = Up(512 // self.channel_down, 256 // factor // self.channel_down, bilinear)
        self.up3 = Up(256 // self.channel_down, 128 // factor // self.channel_down, bilinear)
        self.up4 = Up(128 // self.channel_down, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if not self.return_features:
            x_c = x
            x1 = self.inc(x_c)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits_o = self.outc(x)
            for _ in range(1, self.samples):
                x1 = self.inc(x_c)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x = self.up1(x5, x4)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                logits = self.outc(x)
                logits_o += logits
            logits_o /= self.samples
            return logits_o
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)  
            return x5 

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CycleGAN
class Discriminator(nn.Module):
    def __init__(self, input_nc, classify=False, star_size=1, be=False, embedding_size=None):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.AvgPool2d(2,2),
                    nn.InstanceNorm2d(64), 
                    nn.ReLU()]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.ReLU()]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.ReLU()]

        model += [  nn.Conv2d(256, 512 - (embedding_size if embedding_size is not None else 0), 4, padding=1, stride=2),
                    nn.InstanceNorm2d(512 - (embedding_size if embedding_size is not None else 0)), 
                    nn.ReLU()]

        if embedding_size is not None:
            model_emb = [nn.Linear(embedding_size, embedding_size),
                        nn.LayerNorm(embedding_size), 
                        nn.ReLU()]   

            model_emb += [nn.Linear(embedding_size, embedding_size),
                        nn.LayerNorm(embedding_size), 
                        nn.ReLU()]   
            self.model_emb = nn.Sequential(*model_emb)

        # FCN discrimination layer
        self.disc = nn.Conv2d(512, 1, 4, padding=1)
        if classify:
            # FCN classificationlayer
            self.pred = nn.Sequential(nn.Conv2d(512, 32, 1, padding=0), 
                                      nn.Flatten(),
                                      nn.Linear(288, star_size))

        self.model = nn.Sequential(*model)

        self.classify = classify
        self.star_size= star_size

    def forward(self, x, T=None):
        x =  self.model(x)

        if T is not None:
            T = self.model_emb(T)
            T = T.view(*T.shape, 1,1).expand(*T.shape, *x.shape[2:])
            x = torch.cat((x, T), dim=1)

        x_disc = self.disc(x)
        x_disc = F.avg_pool2d(x_disc, x_disc.size()[2:]).view(x_disc.size()[0], -1)
        if not self.classify:
            return torch.sigmoid(x_disc)
        else:
            x_pred = self.pred(x)
            return x_disc, x_pred


class Embedding3D(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = nn.Parameter(m)
    
    def forward(self, i):
        return self.m[i]

        
class StarGAN(nn.Module):
    def __init__(self, star_size=2, embedding_size=128, many_to_one=False, noisy=False):
        super().__init__()
        self.G_S1_S2 = UNetStyleDown(3, 3, embedding_size=embedding_size)
        self.D1 = Discriminator(3, classify=False, embedding_size=embedding_size)
        self.star_size = star_size
        self.label_embeddings = nn.Embedding(num_embeddings=star_size, embedding_dim=embedding_size)
        self.many_to_one = many_to_one
        self.noisy = noisy

    def S1_S2(self, S1, T):
        return self.G_S1_S2(S1, T, None, None)

    def forward(self, S1, S2, L1, L2):
        T1 = self.label_embeddings(L1)
        T2 = self.label_embeddings(torch.zeros_like(L1))

        if self.noisy:
            noise = np.random.randn(T1.shape[0], T1.shape[1]) * 0.01
            T1 += torch.from_numpy(noise).to(T1.device)

        #  1 2 1
        S2_f = self.S1_S2(S1, T2)
        S1_ff = self.S1_S2(S2_f, T1)

        # 2 1 2
        S1_f = self.S1_S2(S2, T1)
        S2_ff = self.S1_S2(S1_f, T2)

        # D1
        D1_r = self.D1(S1, T1)
        D1_f = self.D1(S1_f, T1)

        # D2
        D2_r = self.D1(S2, T2)
        D2_f = self.D1(S2_f, T2) + self.D1(S2_ff, T2)

        # Identity
        S1_i = self.S1_S2(S1, T2)
        S2_i = self.S1_S2(S2, T1)


        return  S1_i, S2_i, S1_f, S2_f, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f 

class BottleGAN(nn.Module):
    def __init__(self, star_size=2, embedding_size=128, noisy=False):
        super().__init__()
        self.G_S1_S2 = SBNet(channels = 3,embedding_size = embedding_size, hidden_dim=128, linear=False)
        self.G_S2_S1 = SBNet(channels = 3, embedding_size = embedding_size,hidden_dim=128, linear=False)

        self.D1 = Discriminator(3, classify=False, embedding_size=embedding_size)
        self.D2 = Discriminator(3, classify=False)
        self.star_size = star_size
        self.noisy = noisy
        self.label_embeddings = nn.Embedding(num_embeddings=star_size, embedding_dim=embedding_size)

    def S1_S2(self, S1, T, mt, c_ratio):
        return self.G_S1_S2(S1, T, mt, c_ratio)

    def S2_S1(self, S2, T, ms, c_ratio):
        return self.G_S2_S1(S2, T, ms, c_ratio)

    def forward(self, S1, S2, L1, L2):
        T1 = self.label_embeddings(L1)
        mt = None
        ms = None

        c_ratio = None

        if self.noisy:
            noise = np.random.randn(T1.shape[0], T1.shape[1]) * 0.01
            T1 += torch.from_numpy(noise).to(T1.device)

        #  1 2 1
        S2_f = self.S1_S2(S1, T1, mt, c_ratio)
        S1_ff = self.S2_S1(S2_f, T1, ms, c_ratio)

        # 2 1 2
        S1_f = self.S2_S1(S2, T1, ms, c_ratio)
        S2_ff = self.S1_S2(S1_f, T1, mt, c_ratio)

        # D1
        D1_r = self.D1(S1, T1)
        D1_f = self.D1(S1_f, T1)

        # D2
        D2_r = self.D2(S2)
        D2_f = self.D2(S2_f) + self.D2(S2_ff)

        # Identity
        S1_i = self.S2_S1(S1, T1, ms, c_ratio)
        S2_i = self.S1_S2(S2, T1, mt, c_ratio)


        return  S1_i, S2_i, S1_f, S2_f, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f


class StainGAN(nn.Module):
    def __init__(self, be=False, embedding_size=128):
        super().__init__()
        self.G_S1_S2 = SBNet(channels = 3, embedding_size = embedding_size, hidden_dim=128, linear=False)
        self.G_S2_S1 = SBNet(channels = 3, embedding_size = embedding_size, hidden_dim=128, linear=False)
        self.D1 = Discriminator(3, classify=False, be=be)
        self.D2 = Discriminator(3, classify=False, be=be)
        self.noisy = True
        self.label_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=embedding_size)

    def S1_S2(self, S1, T, mt, c_ratio):
        return self.G_S1_S2(S1, T, mt, c_ratio)

    def S2_S1(self, S2, T, ms, c_ratio):
        return self.G_S2_S1(S2, T, ms, c_ratio)

    def forward(self, S1, S2, L1, L2):
        T1 = self.label_embeddings(torch.zeros(S1.shape[0], device=S1.device, dtype=int))
        T2 = self.label_embeddings(torch.ones(S1.shape[0],  device=S2.device, dtype=int))
        mt = None
        ms = None
        c_ratio = None

        if self.noisy:
            noise = np.random.randn(T1.shape[0], T1.shape[1]) * 0.01
            T1 += torch.from_numpy(noise).to(T1.device)

        #  1 2 1
        S2_f = self.S1_S2(S1, T1, mt, c_ratio)
        S1_ff = self.S2_S1(S2_f, T2, ms, c_ratio)

        # 2 1 2
        S1_f = self.S2_S1(S2, T2, ms, c_ratio)
        S2_ff = self.S1_S2(S1_f, T1, mt, c_ratio)

        # D1
        D1_r = self.D1(S1)
        D1_f = self.D1(S1_f)

        # D2
        D2_r = self.D2(S2)
        D2_f = self.D2(S2_f) + self.D2(S2_ff)

        # Identity
        S1_i = self.S2_S1(S1, T2, ms, c_ratio)
        S2_i = self.S1_S2(S2, T1, mt, c_ratio)

        return  S1_i, S2_i, S1_f, S2_f, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f

