"""
Layers design for the neural network
"""
import torch
from torch import Tensor

import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing

from gnn_clrs_reasoning.utils import MLP


class MPGNNConv(MessagePassing):
    """
    Simple layer for message passing
    """

    def __init__(self, node_dim, edge_dim, layers=3):
        super().__init__(aggr="mean", node_dim=0)
        self.lin_edge = MLP(
            in_dim=node_dim * 2 + edge_dim, out_dim=edge_dim, hidden_layers=layers
        )
        self.lin_node = MLP(
            in_dim=node_dim + edge_dim, out_dim=node_dim, hidden_layers=layers
        )

    def forward(self, node, edge_index, edge_attr):
        """
        here we apply the message passing function
        and then we apply the MLPs to the output of the message passing function
        """
        init_node = node

        # message passing
        message_info = self.propagate(edge_index, x=node, edge_attr=edge_attr)

        # we concat the output of the message passing function with the input node features
        node = torch.cat((node, message_info), dim=-1)

        # now we apply the MLPs with residual connections
        node = self.lin_node(node) + init_node

        return node

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor):
        """
        Message function for the message passing
        Basically we concatenate the node features and the edge features
        Args:
            x_j (Tensor): Tensor of shape (E, node_dim) where E is the number of edges. FloatTensor
            x_i (Tensor): Tensor of shape (E, node_dim) where E is the number of edges. FloatTensor
            edge_attr (Tensor): Tensor of shape (E, edge_dim) where E is the number of edges. FloatTensor
        """
        edge_info = torch.cat((x_i, x_j, edge_attr), dim=-1)
        
        #breakpoint()

        edge_info = self.lin_edge(edge_info)

        self.edge_info = edge_info

        return edge_info
