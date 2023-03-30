"""
Utility functions for GNNs
"""

from torch import nn


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

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
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
