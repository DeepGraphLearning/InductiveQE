from collections.abc import Sequence

import torch
from torch import nn
from torchdrug import layers

#from .compgcn_layer import CompGCNConv
from .compgcn_layer_true import TrueCompGCNConv
#from ..layer import CompositionalGraphConv as TrueCompGCNConv


class CompGCN(nn.Module):

    def __init__(self, input_dim=None, hidden_dims=None, edge_input_dim=None,
                 message_func=None, aggregate_func=None, layer_norm=False,
                 short_cut=False, activation="relu", concat_hidden=False, use_boundary=False,
                 use_attention=False, num_heads=4):
        super(CompGCN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        # self.dims = [input_dim] + list(hidden_dims)
        self.dims = list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(TrueCompGCNConv(self.dims[i], self.dims[i + 1], edge_input_dim,
                                           message_func, aggregate_func, activation, layer_norm=layer_norm,
                                           use_boundary=use_boundary, use_attention=use_attention, num_heads=num_heads))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim

        if self.concat_hidden:
            self.mlp = layers.MLP(feature_dim, [feature_dim] * 2)


    def forward(self, graph, input, relation_embs, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        graph.relation_embs = relation_embs
        # we don't use boundaries in compgcn
        # graph.boundary = input

        for layer in self.layers:
            hidden, relation_embs = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden
            graph.relation_embs = relation_embs

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
            node_feature = self.mlp(node_feature)
        else:
            node_feature = hiddens[-1]

        return node_feature, relation_embs