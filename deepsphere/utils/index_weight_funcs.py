"""Functions related to getting the laplacian and the right number of pixels after pooling/unpooling.
"""

import numpy as np
import torch

from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, remove_self_loops

from .get_ico_coords import get_ico_coords
from deepsphere.utils.samplings import (
    icosahedron_nodes_calculator,
    icosahedron_order_calculator,
)

__all__ = ['get_icosahedron_weights']


def get_icosahedron_weights(nodes, depth):
    """Get the icosahedron laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.

    Returns:
        laps (list): increasing list of laplacians.
    """
    edge_list = []
    weight_list = []
    order = icosahedron_order_calculator(nodes)
    for _ in range(depth):
        nodes = icosahedron_nodes_calculator(order)
        order_initial = icosahedron_order_calculator(nodes)
        coords = get_ico_coords(int(order_initial))
        coords = torch.from_numpy(coords)
        edge_index = knn_graph(coords, 6 if order else 5)
        if order:
            dist = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], p=2, dim=1)
            _, extra_idx = torch.topk(dist, 12)
            edge_index[0, extra_idx] = edge_index[1, extra_idx]
            edge_index, _ = remove_self_loops(edge_index)
        edge_list.append(edge_index)
        weight_list.append(None)
        order -= 1
    return edge_list[::-1], weight_list
