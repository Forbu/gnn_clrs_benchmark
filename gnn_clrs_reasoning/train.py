"""
Training module
"""
from torch_geometric.data import DataLoader

import pytorch_lightning as pl

from gnn_clrs_reasoning.models import ProgressiveGNN
from gnn_clrs_reasoning.preprocessing import PreprocessGraphDataset


def train(graph_type="dfs", folder="/tmp/CLRS30"):
    """
    This function is used to train the model
    """

    # init the model
    model = ProgressiveGNN(
        node_dim=2,
        edges_dim=2,
        hidden_dim=128,
        nb_head=4,
        m_iter=5,
        n_iter=5,
        lambda_coef=0.5,
    )

    # init the datasets / dataloaders
    dataloader_train = create_dataloader(folder, graph_type, "train")
    dataloader_eval = create_dataloader(folder, graph_type, "eval")

    # init the trainer (pytorch lightning)
    trainer = pl.Trainer(
        # time 1 hour
        max_time="01:00:00",
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
        ],
        # gradient clipping
        gradient_clip_val=1.0,
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name="progressive_gnn"),
        precision="16-mixed",
    )

    # train the model
    trainer.fit(model, dataloader_train)

    # save the model
    trainer.save_checkpoint("progressive_gnn.ckpt")

    # final operation to evaluate the model on the test set (OOD)
    # TODO: implement this
    # evaluate_progressive_gnn(model, dataloader_eval)


def create_dataloader(folder, algorithm, split):
    """
    This function is used to create the dataloader
    """

    # init the dataset
    dataset = PreprocessGraphDataset(folder, algorithm, split)

    # init the dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # return the dataloader
    return dataloader
