"""
In this module we test the preprocessing of the graph vraible
"""

import torch
from gnn_clrs_reasoning.preprocessing import preprocess_dfs, generate_dataset_graph


def test_preprocess_dfs():
    """
    Test the preprocessing of the dfs graph
    """
    train_ds, num_samples, spec = generate_dataset_graph(
        folder="/tmp/CLRS30", algorithm="dfs", split="train", batch_size=32
    )

    # sample the first graph
    for feedback in train_ds.as_numpy_iterator():
        break

    # now we have to preprocess the graph
    graphs, nb_edges = preprocess_dfs(feedback)

    assert graphs.x.shape == (32 * 16, 1)
    assert graphs.edge_index.shape == (2, torch.sum(nb_edges))
