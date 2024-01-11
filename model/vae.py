import torch
import torch.nn as nn

import math

# from typing import List, Callable, Union, Any, TypeVar, Tuple
from typing import Union
from collections import OrderedDict

def calc_conv_return_shape(dim: tuple[int, int],
                           kernel_size: Union[int, tuple],
                           stride: Union[int, tuple] = (1, 1),
                           padding: Union[int, tuple] = (0, 0),
                           dilation: Union[int, tuple] = (1, 1)) -> tuple[int, int]:
    """
    Calculates the return shape of the Conv2D layer.
    Works on the MaxPool2D layer as well

    See Also: https://github.com/pytorch/pytorch/issues/79512

    See Also: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    See Also: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    Args:
        dim: The dimensions of the input. For instance, an image with (h * w)
        kernel_size: kernel size
        padding: padding size
        dilation: dilation size
        stride: stride size

    Returns:
        Dimensions of the output of the layer
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(stride, int):
        stride = (stride, stride)
    h_out = math.floor(
        (dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
        / stride[0] + 1)
    w_out = math.floor(
        (dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
        / stride[1] + 1)
    return h_out, w_out

## https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/14
class FixedConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(FixedConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv
        
    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x
    
class BaseVAE(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 in_shape: tuple[int, int],
                 latent_dim: int,
                ):
        super().__init__()
        
        vars1 = (3,2,1,1)
        inner_shapes_1 = calc_conv_return_shape(in_shape,*vars1)
        inner_shapes_2 = calc_conv_return_shape(inner_shapes_1,*vars1)
        
        # Encoder Definition
        self.encoder = nn.Sequential(OrderedDict([('block1',
                                                    nn.Sequential(OrderedDict([('conv1', nn.Conv2d(in_channels, 512, kernel_size=3,  stride=2, padding=1, groups=in_channels, bias=False)),
                                                                               ('norm1', nn.InstanceNorm2d(512, affine=True)),
                                                                               ('pwconv1', nn.Conv2d(512, 512, kernel_size=1)),
                                                                               ('act1', nn.GELU()),
                                                                              ]))
                                                  ),
                                                  ('block2',
                                                   nn.Sequential(OrderedDict([('conv1', nn.Conv2d(512, 512, kernel_size=3,  stride=2, padding=1, groups=512, bias=False)),
                                                                              ('norm1', nn.InstanceNorm2d(512, affine=True)),
                                                                              ('pwconv1', nn.Conv2d(512, 512, kernel_size=1)),
                                                                              ('act1', nn.GELU()),
                                                                             ]))
                                                  )
                                                 ]))
        self.flatten = nn.Flatten(start_dim=1)
        
        # Factorized Gaussian Posteriors Definition
        self.lin1_mu = nn.Linear(512*inner_shapes_2[0]*inner_shapes_2[1], latent_dim)
        self.lin1_log_sigma = nn.Linear(512*inner_shapes_2[0]*inner_shapes_2[1], latent_dim)

        # Decoder Definition
        self.lin1_decoder = nn.Linear(latent_dim, 512*inner_shapes_2[0]*inner_shapes_2[1])
        self.unflatten = nn.Unflatten(1, torch.Size([512, inner_shapes_2[0], inner_shapes_2[1]]))
        self.decoder = nn.Sequential(OrderedDict([('block1',
                                                    nn.Sequential(OrderedDict([('tconv1', FixedConvTranspose2d(nn.ConvTranspose2d(512, 512, kernel_size=3,  stride=2, padding=1, groups=512, bias=False), output_size=inner_shapes_1)),
                                                                               ('norm1', nn.InstanceNorm2d(512, affine=True)),
                                                                               ('pwconv1', nn.Conv2d(512, 512, kernel_size=1)),
                                                                               ('act1', nn.GELU()),
                                                                              ]))
                                                  ),
                                                  ('block2',
                                                   nn.Sequential(OrderedDict([('tconv1', FixedConvTranspose2d(nn.ConvTranspose2d(512, in_channels, kernel_size=3,  stride=2, padding=1, groups=in_channels, bias=False), output_size=in_shape)),
                                                                              ('norm1', nn.InstanceNorm2d(in_channels, affine=True)),
                                                                              ('pwconv1', nn.Conv2d(in_channels, in_channels, kernel_size=1)),
#                                                                               ('act1', nn.GELU()),
                                                                              ('act1', nn.Tanh()),
                                                                             ]))
                                                  )
                                                 ]))
        # Latent variable Z hook
        self.latent_z = nn.Identity()
    
    def encode(self,
               x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.flatten(h)
        
        mu = self.lin1_mu(h)
        log_sigma = self.lin1_log_sigma(h)
        
        return (mu, log_sigma)
    
    def decode(self,
               z: torch.Tensor) -> torch.Tensor:
        
        h = self.lin1_decoder(z)
        h = self.unflatten(h)
        h = self.decoder(h)
        
        return h    
    
    def reparameterization_trick(self,
                                 mu: torch.Tensor,
                                 log_sigma: torch.Tensor) -> torch.Tensor:
        
        epsilon = torch.distributions.Normal(0.0, 1.0)
        # https://github.com/pytorch/pytorch/issues/7795
        epsilon.loc = epsilon.loc.cuda()
        epsilon.scale = epsilon.scale.cuda()

        sigma = torch.exp(log_sigma)
        
        z = mu + sigma*epsilon.sample(sigma.shape)
        
        return z
        
    def forward(self,
                x: torch.Tensor):

        mu, log_sigma = self.encode(x)        
        z = self.reparameterization_trick(mu, log_sigma)
        x_proj = self.decode(z)
        
        self.latent_z(z)
        
        return OrderedDict([(f'x_proj', x_proj),
                            (f'mu', mu),
                            (f'log_sigma', log_sigma),
                            ])