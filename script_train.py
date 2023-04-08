"""Script for training a model using mini-batch gradient descent"""

import argparse

import torch

from torch_geometric.data import DataLoader

import lightning as L

from gnn_clrs_reasoning.models import ProgressiveGNN
from gnn_clrs_reasoning.preprocessing import PreprocessGraphDataset


def create_dataloader(folder, algorithm, split, batch_size=32):
    """
    This function is used to create the dataloader
    """

    # init the dataset
    dataset = PreprocessGraphDataset(folder, algorithm, split)

    # init the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # return the dataloader
    return dataloader


if __name__ == "__main__":

    folder = "/tmp/CLRS30"
    graph_type = "dfs"

    # init the model
    model = ProgressiveGNN(
        node_dim=1,
        edges_dim=2,
        hidden_dim=32,
        m_iter=8,
        n_iter=8,
        lambda_coef=0.5,
    )

    # init the datasets / dataloaders
    dataloader_train = create_dataloader(folder, graph_type, "train")
    dataloader_eval = create_dataloader(folder, graph_type, "train")

    # init the trainer (pytorch lightning)
    trainer = L.Trainer(
        # time 1 hour
        max_time={"minutes": 500},
        # gradient clipping
        logger=L.pytorch.loggers.tensorboard.TensorBoardLogger(
            "lightning_logs", name="progressive_gnn"
        ),
        precision="16-mixed",
        # clipping gradients
        gradient_clip_val=1.0,
        limit_val_batches=200,
        val_check_interval=3000,
    )

    # train the model
    trainer.fit(model, dataloader_train, dataloader_eval)

    # save the model
    trainer.save_checkpoint("progressive_gnn_next.pt")

    # save model weights
    torch.save(model.state_dict(), "progressive_gnn_weights_next.pt")
