"""
Module part containing the preprocessing code for the experiments.
"""


import clrs

# classic torch
import torch
from torch import nn

# torch geometric
import torch_geometric
from torch_geometric.data import Data

GRAPH_NAMES = [
    "dfs",
    "bfs",
    "topological_sort",
    "articulation_points",
    "bridges",
    "strongly_connected_components",
    "mst_kruskal",
    "mst_prim",
    "bellman_ford",
    "dijkstra",
    "dag_shortest_paths",
    "floyd_warshall",
]


def generate_dataset_graph(
    folder="/tmp/CLRS30", algorithm="dfs", split="train", batch_size=32
):
    """
    Preprocess the graphs and save them to disk.
    """
    train_ds, num_samples, spec = clrs.create_dataset(
        folder=folder,
        algorithm=algorithm,
        split=split,
        batch_size=batch_size,
    )

    # now we have a lot of work to do to get the data into the right format
    # according to what type of algorithm we are using
    return train_ds, num_samples, spec

def preprocess_dfs(batch_train_ds):
    """
    Preprocess the dfs graph.
    Receive from the dataset the graph and the target
    """

    # first we have to retrieve the graph and the target
    # the graphs are located in the train_ds.inputs.
    graph_sparse = torch.Tensor(batch_train_ds.features.inputs[0].data)

    # A is a tensor of shape (batch_size, num_nodes, num_nodes)
    # for every graph in the batch we have a matrix of shape (num_nodes, num_nodes)
    # we want to transform this matrix into a list of edges
    # of size (num_edges, 2)
    # where num_edges is the number of edges in the graph

    # we first need to get the number of edges in the graph
    # we can do this by summing the number of 1 in the adjacency matrix
    nb_edges = torch.sum(graph_sparse, dim=(1, 2))

    # now for each graph in the batch we have the number of edges
    # we can use this information to create the list of edges
    list_graph = []

    for i in range(graph_sparse.shape[0]):
        # we create a mask to select the edges
        mask = graph_sparse[i] == 1

        # we use the mask to select the edges
        edges = torch.where(mask)
        
        edges = torch.stack(edges, dim=0)

        # we retrive the information about the nodes (node features)
        # the node feature is simply zeros except for the source node (which is the node index at 0)
        nodes_features = torch.zeros((graph_sparse.shape[1], 1))
        nodes_features[0] = 1

        target = torch.tensor(batch_train_ds.outputs[0].data[0])

        graph = torch_geometric.data.Data(
            x=nodes_features,
            edge_index=edges,
            y=target,
        )

        list_graph.append(graph)


    # now we can construct a graph batch
    batch_graphs = torch_geometric.data.Batch.from_data_list(list_graph)
    
    return batch_graphs, nb_edges
