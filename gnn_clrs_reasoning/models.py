"""
Module containing the models used in the experiments.
Those are special models that are used to solve the graph problems.
In some models one have to forecast edges values
"""

# main torch imports
import torch
from torch import nn

from torch_geometric.nn import GATv2Conv

import pytorch_lightning as pl

from gnn_clrs_reasoning.utils import MLP


class ProgressiveGNN(pl.LightningModule):
    """
    Model used to solve the ray tracing problem.
    Progressive GNN is a GNN that is trained in a progressive way.
    The idea is to used the training paradigm in https://arxiv.org/abs/2202.05826
    to compute
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()


class BlockGNN(nn.Module):
    """
    Block GNN is a GNN that is trained in a block wise fashion.
    """

    def __init__(self, nb_layers=3, hidden_dim=128, nb_head=4) -> None:
        super().__init__()
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        self.nb_head = nb_head

        # init the layers
        self.layers_messages = nn.ModuleList()

        for _ in range(self.nb_layers):
            self.layers_messages.append(
                GATv2Conv(
                    hidden_dim,
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
                nodes = self.layers_nodes[i](nodes)
                nodes = self.layers_messages[i](nodes, edge_index, edge_attr)
