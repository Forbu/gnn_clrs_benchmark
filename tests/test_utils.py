"""
Test module for the utils module
"""

import pytest

import torch
from torch import nn

from torch_scatter import scatter_softmax

from gnn_clrs_reasoning.utils import MLP, compute_edges_loss

NB_NODES = 10
NODES_DIM = 1


def test_graph_data():
    """
    Initialize a graph to test the utils functions
    """
    # init nodes
    nodes = torch.randn(NB_NODES, 1)

    # create edges
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9],
            [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 8, 9],
        ]
    )

    # create edges attributes (propability of the edge)
    edge_attr = torch.randn(edge_index.shape[1], 1)

    # we apply a softmax to the edge attributes
    edge_attr = scatter_softmax(edge_attr, edge_index[0], dim=0)

    # we also create a subgraph exemple
    subgraph_edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 0, 1, 4, 5, 4, 7, 6, 7, 8]]
    )

    # now we can apply the loss
    loss_fn = nn.BCELoss()

    loss = compute_edges_loss(
        edge_attr, edge_index.long(), subgraph_edge_index.long(), loss_fn=loss_fn
    )

    assert loss.size() == torch.Size([])
