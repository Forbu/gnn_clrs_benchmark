"""
Test module for the utils module
"""

import pytest

import torch
from torch import nn

from torch_scatter import scatter_softmax

from gnn_clrs_reasoning.utils import MLP, compute_edges_loss, compute_edge_probability

NB_NODES = 10
NODES_DIM = 1


@pytest.fixture(scope="module", autouse=True, name="graph")
def init_graph():
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

    return nodes, edge_index, edge_attr, subgraph_edge_index


def test_graph_data(graph):
    """
    Initialize a graph to test the utils functions
    """
    nodes, edge_index, edge_attr, subgraph_edge_index = graph

    # now we can apply the loss
    loss_fn = nn.BCELoss()

    loss = compute_edges_loss(
        edge_attr, edge_index.long(), subgraph_edge_index.long(), loss_fn=loss_fn
    )

    assert loss.size() == torch.Size([])


def test_compute_edge_probability(graph):
    """
    Test the edge probability computation
    """
    nodes, edge_index, edge_attr, subgraph_edge_index = graph

    # compute the edge probability
    edge_probability = compute_edge_probability(edge_index, subgraph_edge_index)

    assert edge_probability.size() == torch.Size([edge_index.shape[1]])
