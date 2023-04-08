"""
Script to analyse the performance of the model
on out of distribution graphs (OOD).
"""


import argparse
import numpy as np

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
        m_iter=10,
        n_iter=8,
        lambda_coef=0.5,
    )

    # init the datasets / dataloaders
    dataloader_test = create_dataloader(folder, graph_type, "test")

    print(torch.load("progressive_gnn.pt").keys())

    # load the model from .pt file
    model.load_state_dict(torch.load("progressive_gnn.pt")["state_dict"])

    # define dataloader
    print(model)

    model.to("cuda:0")

    # now we want to test the model on out of distribution graphs
    for idx, batch in enumerate(dataloader_test):

        print(batch.ptr)

        batch["nb_iter"] = 20
        batch.cuda()

        loss, accuracy, prediction = model.test_step(batch, idx)
        break

    # now we can save the prediction and batch target to a file
    # get the data
    nodes, edge_index, edge_attr, edge_target = (
        batch["x"].cpu().numpy(),
        batch["edge_index"].cpu().numpy(),
        batch["edge_attr"].cpu().numpy(),
        batch["edge_target"].cpu().numpy(),
    )

    prediction = prediction.cpu().numpy()

    # now we can save the data to a file
    np.savez(
        "prediction.npz",
        nodes=nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_target=edge_target,
        prediction=prediction,
    )

    
    
    