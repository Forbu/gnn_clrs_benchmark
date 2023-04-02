"""
Test module to evaluate models components
"""

import pytest

import torch
from torch import nn

from gnn_clrs_reasoning.models import ProgressiveGNN, BlockGNN, EdgesSoftmax

NODES_DIM = 130
EDGES_DIM = 3
NB_NODES = 200
NB_EDGES = 500

# some fixtures to generate some data (graph data)
@pytest.fixture(scope="module", autouse=True, name="graph_data")
def graph_data():
    """
    Here we initialize some graph data
    """

    nb_nodes = NB_NODES
    nodes_dim = NODES_DIM

    nb_edges = NB_EDGES
    edges_dim = EDGES_DIM

    # init nodes
    nodes = torch.randn(nb_nodes, nodes_dim)

    # init edges
    edge_index = torch.randint(0, nb_nodes, (2, nb_edges))

    # init edges attributes
    edge_attr = torch.randn(nb_edges, edges_dim)

    return nodes, edge_index, edge_attr


@pytest.fixture(scope="module", autouse=True, name="model")
def create_model():
    """
    simple model creation
    """

    block_gnn = BlockGNN(
        nodes_dim=NODES_DIM, edges_dim=EDGES_DIM, hidden_dim=128, nb_head=4
    )

    edge_softmax = EdgesSoftmax(
        nodes_dim=NODES_DIM, edges_dim=EDGES_DIM, nb_layers=2, hidden_dim=128
    )

    return block_gnn, edge_softmax


@pytest.fixture(scope="module", autouse=True, name="progressive_gnn")
def create_progressive_gnn():
    """
    simple model creation
    """

    progressive_gnn = ProgressiveGNN(
        node_dim=NODES_DIM,
        edges_dim=EDGES_DIM,
        hidden_dim=128,
        nb_head=4,
        m_iter=5,
        n_iter=5,
        lambda_coef=0.5,
    )

    return progressive_gnn


def test_block_gnn(model, graph_data):
    """
    Test the block GNN
    """

    block_gnn, _ = model
    nodes, edge_index, edge_attr = graph_data

    # now we can forward the model
    nodes = block_gnn(nodes, edge_index, edge_attr)

    assert nodes.shape == (NB_NODES, 128)


def test_edges_softmax(model, graph_data):
    """
    Test the edges softmax
    """

    _, edge_softmax = model
    nodes, edge_index, edge_attr = graph_data

    # now we can forward the model
    edges = edge_softmax(nodes, edge_index, edge_attr)

    assert edges.shape == (NB_EDGES, 1)


def test_progressive_gnn(progressive_gnn, graph_data):
    """
    Test the progressive GNN
    """

    nodes, edge_index, edge_attr = graph_data

    # now we can forward the model
    edges_softmax = progressive_gnn(
        nb_iter=2,
        nodes=nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        progressive=False,
        progressive_iter=0,
    )

    assert edges_softmax.shape == (NB_EDGES, 1)

    # now we can forward the model
    edges_softmax = progressive_gnn(
        nb_iter=2,
        nodes=nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        progressive=True,
        progressive_iter=2,
    )

    assert edges_softmax.shape == (NB_EDGES, 1)


def test_progressive_gnn_train_step(progressive_gnn, graph_data):
    """
    Test the progressive GNN
    """

    nodes, edge_index, edge_attr = graph_data

    batch = {
        "x": nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_target": torch.randint(0, 2, (NB_EDGES, 1)),
    }

    # now we can use the training_step
    loss = progressive_gnn.training_step(batch, 0)

    assert loss.size() == torch.Size([])
