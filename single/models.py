from torch import nn
from torch.nn import functional as F

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Upsampling(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128, 3], kernel_size=4, stride=2, padding=1, activation=nn.PReLU, batchnorm=True, input_resolution=None, out_resolution=(225,225)):
        """Standard upconvolution network.

        Args:
            layers (list, optional): Size of layers. Defaults to [1280, 512, 256, 128, 3]
            kernel_size (int, optional): Size of kernel. Defaults to 5.
            stride (int, optional): Size of stride. Defaults to 2.
            padding (int, optional): Size of padding. Defaults to 2.
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
            input_resolution ((int, int), optional): Reshape flat input. Set to None if not needed. Defaults to (7,7).
            out_resolution ((int, int), optional): Final resizing layer, may be handy for slight deviations. Set to None if not needed. Defaults to (225,225).
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.out_resolution = out_resolution
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=layers[i - 1], out_channels=layers[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        if self.input_resolution is not None:
            x = x.view(x.shape[0], -1, self.input_resolution[0], self.input_resolution[1])
        x = self.model(x)
        if self.out_resolution is not None:
            x = F.interpolate(x, self.out_resolution)
        return x


class ReshapeChannels(nn.Module):
    def __init__(self, backbone=None, in_channels=1280, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.PReLU, groups=1, flat=True, endpoints=False):
        """Convolutional bottleneck (single layer)

        Args:
            backbone (torch.nn.Module, optional): ConvNet to decorate. Defaults to None.
            in_channels (int, optional): Number of input channels. Defaults to 1280.
            out_channels (int, optional): Number of output channels. Defaults to 64.
            kernel_size (int, optional): Size of kernel. Defaults to 3.
            stride (int, optional): Size of stride. Defaults to 1.
            padding (int, optional): Size of padding. Defaults to 1.
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
            groups (int, optional): Number of equally sized groups. Defaults to 1.
            flat (bool, optional): Wether to flat output. Defaults to True.

        Raises:
            NotImplementedError: Works only on top of another ConvNet
        """
        super().__init__()
        if backbone is None:
            raise NotImplementedError(
                "You need to specify a backbone network.")
        self.backbone = backbone
        self.model = nn.Sequential(nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups), activation())
        self.flat = flat
        self.endpoints = endpoints

    def forward(self, x):
        if self.endpoints:
            x = self.backbone(x, endpoints=True)
            return self.model(x[-1]), x
        else:
            x = self.backbone(x)
            if self.flat:
                return self.model(x).reshape(x.shape[0], -1)
            else:
                return self.model(x)


class CombinedNet(nn.Module):
    def __init__(self, backbone=None, predictor=None, distributed=False):
        """Main building block of Super Selfish. Combines backbone features with a prediction head.
        Args:
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None.
            distributed (bool, optional): Wether to use nn.DataParallel, also handled in supervisor. Defaults to False.
        Raises:
            NotImplementedError: Backbone and Precitor must be specified.
        """
        super().__init__()
        if backbone is None or predictor is None:
            raise NotImplementedError(
                "You need to specify a backbone and a predictor network.")
        self.backbone = backbone
        self.predictor = predictor

    
    def forward(self, x, *kwargs):
        x = self.backbone(x, *kwargs)
        return self.predictor(x)


class CombinedUNet(nn.Module):
    def __init__(self, backbone=None, predictor=None, unet=None):
        """Main building block of Super Selfish. Combines backbone features with a prediction head.
        Args:
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None.
            distributed (bool, optional): Wether to use nn.DataParallel, also handled in supervisor. Defaults to False.
        Raises:
            NotImplementedError: Backbone and Precitor must be specified.
        """
        super().__init__()
        if backbone is None or predictor is None or unet is None:
            raise NotImplementedError(
                "You need to specify a backbone and a predictor network and a unet networkt.")
        self.backbone = backbone
        self.predictor = predictor
        self.unet = unet

    def forward(self, x):
        x, residuals = self.backbone(x)
        up_residuals = self.unet(residuals)
        return self.predictor(x), up_residuals
