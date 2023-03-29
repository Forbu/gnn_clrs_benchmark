"""
Module containing the models used in the experiments.
Those are special models that are used to solve the graph problems.
In some models one have to forecast edges values
"""

# main torch imports
import torch

import pytorch_lightning as pl



class RayTracingModelGAT(pl.LightningModule):
    """
    Model used to solve the ray tracing problem.
    """

    def __init__(self, hparams):
        """
        Initialize the model.
        """
        super().__init__()
        self.hparams = hparams

        # get the blocks
        self.blocks_encoding_decoding, self.blocks_message_passing = get_blocks_encoding_decoding(
            hparams
        )
        self.blocks_message_passing = get_blocks_message_passing(hparams)

        # get the output layer
        self.output_layer = torch.nn.Linear(hparams["hidden_dim"], 1)

    def forward(self, graph):
        """
        Forward pass of the model.
        """
        # first we have to encode the graph
        x = self.blocks_encoding_decoding(graph.x, graph.edge_index)

        # now we have to pass the message
        x = self.blocks_message_passing(x, graph.edge_index)

        # now we have to decode the graph
        x = self.blocks_encoding_decoding(x, graph.edge_index, reverse=True)

        # now we have to predict the output
        x = self.output_layer(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.
        """
        # get the graph
        graph = batch

        # get the output
        output = self(graph)

        # compute the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, graph.y
        )

        # log the loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.
        """
        # get the graph
        graph = batch

        # get the output
        output = self(graph)

        # compute the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, graph.y
        )

        # log the loss
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step of the model.
        """
        # get the graph
        graph = batch

        # get the output
        output = self(graph)

        # compute the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, graph.y
        )

        # log the loss
        self.log("test_loss", loss)