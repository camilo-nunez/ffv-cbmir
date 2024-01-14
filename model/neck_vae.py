from .vae import BaseVAE
from collections import OrderedDict

import torch
import torch.nn as nn

class NeckVAE(nn.Module):
    def __init__(self,
                 num_vae: int,
                 in_channels: int,
                 in_shape: tuple[int, int],
                 latent_dim: int,
                 ):
        super().__init__()
        
        self.vaes = nn.ModuleList([BaseVAE(in_channels, in_shape, latent_dim)
                                   for i in range(num_vae)])

    def forward(self, inputs):

        if isinstance(inputs, OrderedDict): inputs = list(inputs.values())
        
        x_proj_l = []
        for i, input_l in enumerate(inputs):
            x_proj_l.append(self.vaes[i](input_l))
        
        return OrderedDict([(f'x_dict_l{i}', x_proj_l[i]) for i in range(len(x_proj_l))])