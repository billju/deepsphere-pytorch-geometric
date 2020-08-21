"""Chebyshev convolution layer. For the moment taking as-is from Michaël Defferrard's implementation. For v0.15 we will rewrite parts of this layer.
"""
# pylint: disable=W0221

import math

import torch
from torch import nn
from torch_sparse import SparseTensor
from torch_geometric.nn import ChebConv


__all__ = ['DenseChebConv']

class DenseChebConv(ChebConv):
    def forward(self, x, edge_index, edge_weight=None, lambda_max=None, **kwargs):
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        x = x.unsqueeze(0) if x.dim() == 2 else x
        B, N, _ = x.size()

        edge_index, norm = self.__norm__(edge_index, N,
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype)

        adj_size = (N, N)
        L_hat = SparseTensor.from_edge_index(edge_index, edge_weight,
                                             sparse_sizes=adj_size)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = L_hat @ x
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = L_hat @ Tx_1
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

if __name__ == '__main__':
    m = DenseChebConv(1, 1, 2)

    i = torch.LongTensor([[0, 1, 1, 1],
                          [2, 0, 2, 1]])
    v = torch.FloatTensor([3, 4, 5, 1])[None, :, None]

    tmp = SparseTensor.from_edge_index(i, v.squeeze(), sparse_sizes=(3, 3))
    print((torch.randn(3, 3) @ torch.randn(2, 3, 6)).shape)

    w = torch.FloatTensor([1, 1, 1, 2])

    y = m(v, i, w)
