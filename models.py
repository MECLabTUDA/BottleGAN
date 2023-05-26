import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Adaptor for BottleGAN to super selfish => could be simplified
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Possible Predictors besides nn.idenity
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
class MultiArgsIdentity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        return args


class TemperatureScaledPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5, requires_grad=True)

    def forward(self, x):
        return  x / self.T

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# BottleGAN
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Generator: Style Bottle 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CycleGAN
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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