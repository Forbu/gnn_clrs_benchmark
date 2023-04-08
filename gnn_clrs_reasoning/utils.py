"""
Utility functions for GNNs
"""
import torch
from torch import nn

from torch import vmap


class MLP(nn.Module):
    """
    Simple MLP (multi-layer perceptron)
    """

    # MLP with LayerNorm
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):
        """
        MLP
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', ...
        """

        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, vector):
        """
        Simple forward pass
        """
        return self.model(vector)


def compute_edge_probability(edges_index, subgraph_edges_index):
    """
    We have two sets of edges :
    edges_index is the index of the edges in the graph size (2, nb_edges)
    subgraph_edges_index is the index of the edges in the subgraph size (2, nb_nodes)
    
    we want to compute the probability of the edges in the graph
    value 1 : the edge is in the subgraph
    value 0 : the edge is not in the subgraph
    
    """
    # get the index of the subgraph edges in the edges_index
    index_edges_subgraph = find_index_edges_subgraph(edges_index, subgraph_edges_index)

    # initialize the probability of the edges
    edges_proba = torch.zeros(edges_index.shape[1]).to(edges_index.device)

    # put the probability of the edges in the subgraph to 1
    edges_proba[index_edges_subgraph] = 1

    return edges_proba

def compute_edges_loss(edges_proba, edges_index, subgraph_edges_index, loss_fn):
    """
    Compute the loss over the edges

    params: edges_index is the index of the edges in the graph size (2, nb_edges)
    params: subgraph_edges_index is the index of the edges in the subgraph size (2, nb_nodes)
            the subgraph is of form (2, nb_nodes) and
            the first row is the index of the nodes (0, 1, 2, 3, ...)

    params: edges_proba is the probability of the edges size (nb_edges)

    Every edge in the subgraph is supposed to have a probability of 1 and every other edge
    is supposed to have a probability of 0.
    We use the cross entropy loss to compute the loss over the edges.

    So we have two steps :
    - first step is to compute the index of the edges in the subgraph
    for exemple if we have the edges_index = [[0, 1, 2, 3], [1, 2, 3, 0]]
    and the subgraph_edges_index = [[0, 1], [1, 2]]
    we want to have the index of the edges in the subgraph
    so we have index_edges_subgraph = [0, 1]

    - second step is to compute the index of the edges not in the subgraph
    we can just take the opposite of the previous index
    index_edges_not_subgraph = torch.arange(edges_index.shape[1])[~index_edges_subgraph]
    """

    # get the index of the subgraph edges in the edges_index
    index_edges_subgraph = find_index_edges_subgraph(edges_index, subgraph_edges_index)

    # get the index of the edges not in the subgraph
    init_index_edges_not_subgraph = torch.arange(edges_index.shape[1]).to(
        edges_proba.device
    )

    isin_edge_subgraph = torch.isin(init_index_edges_not_subgraph, index_edges_subgraph)
    index_edges_not_subgraph = init_index_edges_not_subgraph[~isin_edge_subgraph]

    # get the probability of the edges in the subgraph and not in the subgraph
    edges_proba_subgraph = edges_proba[index_edges_subgraph]
    edges_proba_not_subgraph = edges_proba[index_edges_not_subgraph]

    # compute the loss
    edges_label_subgraph = torch.ones_like(edges_proba_subgraph)
    edges_label_not_subgraph = torch.zeros_like(edges_proba_not_subgraph)

    edges_loss_subgraph = loss_fn(edges_proba_subgraph, edges_label_subgraph)
    edges_loss_not_subgraph = loss_fn(
        edges_proba_not_subgraph, edges_label_not_subgraph
    )

    return edges_loss_subgraph + edges_loss_not_subgraph


def find_index_edges_subgraph(edges_index, subgraph_edges_index):
    """
    Function taken from GPT4 with love :)
    params: edges_index is the index of the edges in the graph size (2, nb_edges)
    params: subgraph_edges_index is the index of the edges in the subgraph size (2, nb_nodes)

    return: index_edges_subgraph is the index of the edges in the subgraph size (nb_edges)
            size (nb_nodes)
    """

    def compare_edges(edge_depart, edge_end, subgraph_edge_depart, subgraph_edge_end):
        return (edge_depart == subgraph_edge_depart) & (edge_end == subgraph_edge_end)

    compare_edges_vmap = torch.vmap(
        compare_edges, in_dims=(0, 0, None, None), out_dims=0
    )
    matched_edges = compare_edges_vmap(
        edges_index[0].unsqueeze(1),
        edges_index[1].unsqueeze(1),
        subgraph_edges_index[0],
        subgraph_edges_index[1],
    )
    index_edges_subgraph = torch.nonzero(matched_edges.any(dim=1)).flatten()

    return index_edges_subgraph


