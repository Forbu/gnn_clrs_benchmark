"""
Module containing the models used in the experiments.
Those are special models that are used to solve the graph problems.
In some models one have to forecast edges values
"""

# main torch imports
import torch
from torch import nn

from torch_geometric.nn import GATv2Conv

from torch_scatter.composite import scatter_softmax, scatter_log_softmax

import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from gnn_clrs_reasoning.utils import MLP, compute_edges_loss


class ProgressiveGNN(pl.LightningModule):
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
        nb_head=4,
        m_iter=5,
        n_iter=5,
        lambda_coef=0.5,
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.edges_dim = edges_dim
        self.hidden_dim = hidden_dim
        self.nb_head = nb_head
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
            nodes_dim=hidden_dim + node_dim,
            nb_layers=2,
            hidden_dim=hidden_dim,
            nb_head=nb_head,
            edges_dim=hidden_dim,
        )

        # final layer, the final layer give a softmax over the edges
        # (each nodes give a softmax over the edges)
        self.final_layer = EdgesSoftmax(
            nodes_dim=128, edges_dim=128, nb_layers=2, hidden_dim=128
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

                nodes = self.block_gnn(nodes_input, edge_index, edge_attr)

            # now we can forward the final layer
            edges = self.final_layer(nodes, edge_index, edge_attr)

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
                    nodes = self.block_gnn(nodes_input, edge_index, edge_attr)

            nodes = nodes.detach()  # we detach the nodes to avoid gradient computation

            for _ in range(progressive_iter):
                # now we can forward the final layer
                nodes_input = torch.cat([nodes, nodes_init], dim=-1)
                # now we can forward the block GNN
                nodes = self.block_gnn(nodes_input, edge_index, edge_attr)

            # now we can forward the final layer
            edges = self.final_layer(nodes, edge_index, edge_attr)

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
        edges_softmax = self(
            nb_iter=n_step,
            nodes=nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            progressive=False,
        )

        # compute the loss
        loss_standard = self.loss_fn(
            edges_softmax.squeeze(), edge_target.float().squeeze()
        )

        # now we compute the progressive training
        edges_softmax_progressive = self(
            nb_iter=n_step,
            nodes=nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            progressive=False,
            progressive_iter=0,
        )

        loss_progressive = self.loss_fn(
            edges_softmax_progressive.squeeze(), edge_target.float().squeeze()
        )

        self.log("loss_standard_train", loss_standard)
        self.log("loss_progressive_train", loss_progressive)

        # we also want to log accuracy for the standard and progressive training
        # we compute the accuracy
        acc_standard = self.accuracy_metric(edges_softmax.squeeze(), edge_target)
        acc_progressive = self.accuracy_metric(
            edges_softmax_progressive.squeeze(), edge_target
        )

        # we log the accuracy
        self.log("acc_standard_train", acc_standard)
        self.log("acc_progressive_train", acc_progressive)

        # we also want to log precision and recall for the standard and progressive training
        # we compute the precision
        precision_standard = self.precision_metric(edges_softmax.squeeze(), edge_target)
        precision_progressive = self.precision_metric(
            edges_softmax_progressive.squeeze(), edge_target
        )

        # we log the precision
        self.log("precision_standard_train", precision_standard)
        self.log("precision_progressive_train", precision_progressive)

        # we compute the recall
        recall_standard = self.recall_metric(edges_softmax.squeeze(), edge_target)
        recall_progressive = self.recall_metric(
            edges_softmax_progressive.squeeze(), edge_target
        )

        # we log the recall
        self.log("recall_standard_train", recall_standard)
        self.log("recall_progressive_train", recall_progressive)

        # log the first edge softmax (sigmoid) value
        print(nn.functional.sigmoid(edges_softmax[0:16, 0]))

        print(edge_target[0:16])

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


class BlockGNN(nn.Module):
    """
    Block GNN is a GNN that is trained in a block wise fashion.
    The nodes input size of the GNN is 2*hidden_dim
    and the node output size is hidden_dim.
    """

    def __init__(
        self, nodes_dim=130, nb_layers=2, hidden_dim=128, nb_head=4, edges_dim=2
    ) -> None:
        super().__init__()
        self.nodes_dim = nodes_dim
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        self.nb_head = nb_head

        assert hidden_dim % nb_head == 0, "hidden_dim must be divisible by nb_head"
        assert nodes_dim >= hidden_dim, "nodes_dim must be >= hidden_dim"

        # init the layers
        self.layers_messages = nn.ModuleList()

        for i in range(self.nb_layers):
            if i == 0:
                self.layers_messages.append(
                    GATv2Conv(
                        nodes_dim,
                        hidden_dim // nb_head,
                        heads=nb_head,
                        concat=True,
                        edge_dim=edges_dim,
                    )
                )
            else:
                self.layers_messages.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // nb_head,
                        heads=nb_head,
                        concat=True,
                        edge_dim=edges_dim,
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

        return nodes


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
        # result = scatter_log_softmax(
        #     message[:, 0], edge_index[0, :]
        # )  # shape (nb_edges, 1)
        result = message[:, 0].reshape(-1, 1)

        return result
