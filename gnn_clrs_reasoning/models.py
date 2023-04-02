"""
Module containing the models used in the experiments.
Those are special models that are used to solve the graph problems.
In some models one have to forecast edges values
"""
import time

# main torch imports
import torch
from torch import nn

from torch_geometric.nn import GATv2Conv

from torch_scatter.composite import scatter_softmax, scatter_log_softmax

import lightning as L

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from gnn_clrs_reasoning.utils import MLP
from gnn_clrs_reasoning.layers import MPGNNConv


class ProgressiveGNN(L.LightningModule):
    """
    Model used to solve the ray tracing problem.
    Progressive GNN is a GNN that is trained in a progressive way.
    The idea is to used the training paradigm in https://arxiv.org/abs/2202.05826
    to compute
    """

    def __init__(
        self,
        node_dim=2,
        edges_dim=2,
        hidden_dim=128,
        m_iter=5,
        n_iter=5,
        lambda_coef=0.5,
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.edges_dim = edges_dim
        self.hidden_dim = hidden_dim
        self.m_iter = m_iter
        self.n_iter = n_iter
        self.lambda_coef = lambda_coef

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
            hidden_dim=hidden_dim + node_dim,
            nb_layers=1,
            edges_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        # final layer, the final layer give a softmax over the edges
        # (each nodes give a softmax over the edges)
        self.final_layer = MLP(
            in_dim=hidden_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            norm_type=None,
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.accuracy_metric = BinaryAccuracy()
        self.precision_metric = BinaryPrecision()
        self.recall_metric = BinaryRecall()

    def forward(
        self,
        nb_iter,
        nodes,
        edge_index,
        edge_attr,
        progressive=False,
        progressive_iter=0,
    ):
        """
        Forward pass of the model

        params:
            nb_iter: number of iteration of the block GNN for the classic block
            nodes: nodes features
            edge_index: edges index
            edge_attr: edges features
            progressive: if true we use the progressive training paradigm
            progressive_iter: number of iteration of the block GNN for the progressive training

        return:
            edges: edges values (softmax over the edges)
        """

        if not progressive:

            nodes_init = nodes.clone()

            # first we encode the nodes and edges
            nodes = self.node_encoder(nodes)
            edge_attr = self.edge_encoder(edge_attr)

            for _ in range(nb_iter):
                nodes_input = torch.cat([nodes, nodes_init], dim=-1)
                # now we can forward the block GNN

                nodes, edge_attr = self.block_gnn(nodes_input, edge_index, edge_attr)

            # now we can forward the final layer
            edges = self.final_layer(edge_attr)

            return edges
        else:
            with torch.no_grad():
                nodes_init = nodes.clone()

                # first we encode the nodes and edges
                nodes = self.node_encoder(nodes)
                edge_attr = self.edge_encoder(edge_attr)

                for _ in range(nb_iter):
                    nodes_input = torch.cat([nodes, nodes_init], dim=-1)
                    # now we can forward the block GNN
                    nodes, edge_attr = self.block_gnn(
                        nodes_input, edge_index, edge_attr
                    )

            nodes = nodes.detach()  # we detach the nodes to avoid gradient computation

            for _ in range(progressive_iter):
                # now we can forward the final layer
                nodes_input = torch.cat([nodes, nodes_init], dim=-1)
                # now we can forward the block GNN
                nodes, edge_attr = self.block_gnn(nodes_input, edge_index, edge_attr)

            # now we can forward the final layer
            edges = self.final_layer(edge_attr)

            return edges

    def training_step(self, batch, batch_idx):
        """
        Training step
        """

        # we check if edge_attr is in the batch
        # if not we create it
        if "edge_attr" not in batch:
            batch["edge_attr"] = torch.ones(
                (batch["edge_index"].shape[1], self.edges_dim)
            ).to(batch["edge_index"].device)

        # get the data
        nodes, edge_index, edge_attr, edge_target = (
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["edge_target"],
        )

        # we mix the progressive and classic training
        # we select a random number of iteration for the progressive training
        # and a random number of iteration for the classic training
        n_step = torch.randint(1, self.m_iter, (1,)).item()
        k = torch.randint(1, self.m_iter + 1 - n_step, (1,)).item()

        # we first compute the standard training
        edge_values = self(
            nb_iter=n_step,
            nodes=nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            progressive=False,
        )

        # compute the loss
        # loss_standard = self.loss_fn(
        #     edge_values.squeeze(), edge_target.float().squeeze()
        # )
        loss_standard, edge_values_new = compute_custom_loss(
            edge_values.squeeze(), edge_target.float().squeeze(), edge_index
        )

        # now we compute the progressive training
        edge_values_progressive = self(
            nb_iter=n_step,
            nodes=nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            progressive=True,
            progressive_iter=k,
        )

        # loss_progressive = self.loss_fn(
        #     edge_values_progressive.squeeze(), edge_target.float().squeeze()
        # )
        loss_progressive, edge_values_progr = compute_custom_loss(
            edge_values_progressive.squeeze(), edge_target.float().squeeze(), edge_index
        )

        self.log("loss_standard_train", loss_standard)
        self.log("loss_progressive_train", loss_progressive)

        # we also want to log accuracy for the standard and progressive training
        # we compute the accuracy
        acc_standard = self.accuracy_metric(
            edge_values_new.squeeze(), edge_target.squeeze()
        )
        acc_progressive = self.accuracy_metric(
            edge_values_progr.squeeze(), edge_target.squeeze()
        )

        # we log the accuracy
        self.log("acc_standard_train", acc_standard)
        self.log("acc_progressive_train", acc_progressive)

        # log the first edge softmax (sigmoid) value
        print(edge_values_new[0:16])
        print(edge_target[0:16])

        # we replace the losses by nan_to_num to avoid nan values
        loss_standard = torch.nan_to_num(loss_standard, nan=0.0, posinf=0.0, neginf=0.0)
        loss_progressive = torch.nan_to_num(loss_progressive, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            self.lambda_coef * loss_standard + (1 - self.lambda_coef) * loss_progressive
        )

    def validation_step(self, batch, batch_idx):
        """
        In the validation step we only use the progressive training paradigm
        TODO
        """
        pass

    def configure_optimizers(self):
        """
        Classic Adam optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class BlockGNN(nn.Module):
    """
    Block GNN is a GNN that is trained in a block wise fashion.
    The nodes input size of the GNN is 2*hidden_dim
    and the node output size is hidden_dim.
    """

    def __init__(
        self, nb_layers=2, hidden_dim=128, edges_dim=2, output_dim=128
    ) -> None:
        super().__init__()
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim

        # init the layers
        self.layers_messages = nn.ModuleList()

        for _ in range(self.nb_layers):
            self.layers_messages.append(
                MPGNNConv(node_dim=hidden_dim, edge_dim=edges_dim, layers=2)
            )

        self.final_layer = MLP(
            in_dim=hidden_dim,
            out_dim=output_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, nodes, edge_index, edge_attr):
        """
        Forward pass of the GNN
        """
        for i in range(self.nb_layers):

            nodes = self.layers_messages[i](nodes, edge_index, edge_attr)
            edge_attr = self.layers_messages[i].edge_info

        nodes = self.final_layer(nodes)

        return nodes, edge_attr


def compute_custom_loss(
    edge_values: torch.Tensor, edge_target: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute the custom loss
    """
    # scatter_softmax over the edges
    edge_values = scatter_softmax(
        edge_values, edge_index[1], dim=0
    )  # probalities of the edges

    # now we can compute the loss (custom BCE)
    loss = -torch.mean(
        edge_target * torch.log(edge_values + 1e-6)
        + (1 - edge_target) * torch.log(1 - edge_values + 1e-6)
    )

    return loss, edge_values
