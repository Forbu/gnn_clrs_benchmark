"""
In this module we test the preprocessing of the graph vraible
"""
import pytest

import torch
from gnn_clrs_reasoning.preprocessing import (
    preprocess_dfs,
    generate_dataset_graph,
    PreprocessGraphDataset,
)


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

    assert graphs.x.shape == (32 * 16, 2)
    assert graphs.edge_index.shape == (2, torch.sum(nb_edges))


@pytest.fixture(scope="module", autouse=True, name="dataset")
def create_dataset():
    iter_dataset = PreprocessGraphDataset(
        folder="/tmp/CLRS30", algorithm="dfs", split="train"
    )
    return iter_dataset


def test_preprocess_graph_dataset(dataset):
    """
    Test the preprocessing of the graph dataset
    """

    # sample the first graph
    for idx, graph in enumerate(dataset):
        if idx > 100:
            break

    assert graph.x.shape == (16, 2)


@pytest.mark.benchmark(group="preprocess_graph_dataset", warmup=True, warmup_iterations=3)
def test_preprocess_graph_dataset_benchmark(dataset, benchmark):
    """
    Benchmark the preprocessing of the graph dataset
    """

    result = benchmark(test_preprocess_graph_dataset, dataset)


