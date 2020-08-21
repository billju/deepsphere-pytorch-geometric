"""Layers used in both Encoder and Decoder.
"""
# pylint: disable=W0221
import torch
from torch import nn
from deepsphere.layers.chebyshev import *


class SphericalChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, edge_index, edge_weight,
                 laplacian_type):
        super().__init__()
        assert laplacian_type in ['normalized', 'combinatorial'], 'Invalid normalization'

        self.register_buffer("edge_index", edge_index)
        if edge_weight is None:
            setattr(self, 'edge_weight', None)
        else:
            self.register_buffer("edge_weight", edge_weight)

        self.register_buffer('lambda_max', torch.tensor(2., dtype=torch.float32))
        self.chebconv = DenseChebConv(in_channels, out_channels, kernel_size,
                                      normalization='sym' if laplacian_type == 'normalized' else None)


    def forward(self, x):
        x = self.chebconv(x, self.edge_index, self.edge_weight, self.lambda_max)
        return x


class SphericalChebBN(nn.Module):
    """Building Block with a Chebyshev Convolution, Batchnormalization, and ReLu activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, kernel_size,
                                                **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.transpose(1, 2)).relu()
        return x.transpose(1, 2)


class SphericalChebBNPool(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, pooling, kernel_size, **kwargs):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb_bn = SphericalChebBN(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb_bn(x)
        return x
