"""
Module containing the models used in the experiments.
Those are special models that are used to solve the graph problems.
In some models one have to forecast edges values
"""

# main torch imports
import torch
from torch import nn

from torch_geometric.nn import GATv2Conv

from torch_scatter.composite import scatter_softmax

import pytorch_lightning as pl

from gnn_clrs_reasoning.utils import MLP


class ProgressiveGNN(pl.LightningModule):
    """
    Model used to solve the ray tracing problem.
    Progressive GNN is a GNN that is trained in a progressive way.
    The idea is to used the training paradigm in https://arxiv.org/abs/2202.05826
    to compute
    """

    def __init__(self, node_dim=2, edges_dim=2, hidden_dim=128, nb_head=4) -> None:
        super().__init__()

        # simple node encoder and edge encoder
        self.node_encoder = MLP(
            in_dim=node_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

        self.edge_encoder = MLP(
            in_dim=edges_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

        # now we can create the blockGNN
        self.block_gnn = BlockGNN(
            nodes_dim=hidden_dim + node_dim,
            nb_layers=2,
            hidden_dim=hidden_dim,
            nb_head=nb_head,
        )

        # final layer, the final layer give a softmax over the edges
        # (each nodes give a softmax over the edges)


class BlockGNN(nn.Module):
    """
    Block GNN is a GNN that is trained in a block wise fashion.
    The nodes input size of the GNN is 2*hidden_dim
    and the node output size is hidden_dim.
    """

    def __init__(self, nodes_dim=130, nb_layers=2, hidden_dim=128, nb_head=4) -> None:
        super().__init__()
        self.nodes_dim = nodes_dim
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        self.nb_head = nb_head

        assert hidden_dim % nb_head == 0, "hidden_dim must be divisible by nb_head"
        assert nodes_dim >= hidden_dim, "nodes_dim must be >= hidden_dim"

        # init the layers
        self.layers_messages = nn.ModuleList()

        for _ in range(self.nb_layers):
            self.layers_messages.append(
                GATv2Conv(
                    nodes_dim,
                    hidden_dim // nb_head,
                    heads=nb_head,
                    concat=True,
                    edge_dim=hidden_dim // nb_head,
                )
            )

        self.layers_nodes = nn.ModuleList()
        for _ in range(self.nb_layers):
            self.layers_nodes.append(
                MLP(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                )
            )

        def forward(self, nodes, edge_index, edge_attr):
            """
            Forward pass of the GNN
            """
            for i in range(self.nb_layers):
                nodes = self.layers_messages[i](nodes, edge_index, edge_attr)
                nodes = self.layers_nodes[i](nodes)


class EdgesSoftmax(nn.Module):
    """
    This layer is used to compute the softmax over the edges.
    Basicly we have N_nodes and each nodes have N_edges.

    We have 2 steps :
    - first step is message : e_edge = f(nodes_1, nodes_2, edge_attr)
    - second step is reduce : e_edge = scatter_softmax(e_edge, edge_index[0])
    """

    def __init__(
        self,
        nodes_dim=130,
        edges_dim=128,
        nb_layers=2,
        hidden_dim=128,
    ) -> None:
        super().__init__()

        self.message_layer = MLP(
            in_dim=2 * nodes_dim + edges_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=nb_layers,
        )

    def forward(self, nodes, edge_index, edge_attr):
        """
        Forward pass of the GNN
        2 steps :
        - first step is message : e_edge = f(nodes_1, nodes_2, edge_attr)
        - second step is reduce : e_edge = scatter_softmax(e_edge, edge_index[0])
        """
        nodes_1 = nodes[edge_index[0]]
        nodes_2 = nodes[edge_index[1]]

        inputs_message = torch.cat([nodes_1, nodes_2, edge_attr], dim=1)
        message = self.message_layer(inputs_message)  # shape (nb_edges, 1)

        # use scatter_softmax to compute the softmax over the edges
        # we have to use the edge_index[0] to compute the softmax
        # because each nodes have N_edges
        result = scatter_softmax(message, edge_index[0])  # shape (nb_edges, 1)

        return result
