import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers
from torchdrug.layers import functional


class TrueCompGCNConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, edge_input_dim, message_func="distmult",
                 aggregate_func="pna", activation="relu", layer_norm=False, use_boundary=False,
                 use_attention=False, num_heads=4):
        super(TrueCompGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.use_boundary = use_boundary

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 12, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        self.relation_update = nn.Linear(input_dim, input_dim)

        # CompGCN weight matrices
        self.w_in = nn.Parameter(torch.empty(input_dim, input_dim))
        self.w_out = nn.Parameter(torch.empty(input_dim, input_dim))
        self.w_loop = nn.Parameter(torch.empty(input_dim, input_dim))
        nn.init.xavier_uniform_(self.w_in)
        nn.init.xavier_uniform_(self.w_out)
        nn.init.xavier_uniform_(self.w_loop)
        # self.register_buffer("w_in", nn.Parameter(input_dim, input_dim))
        # self.register_buffer("w_out", nn.Parameter(input_dim, input_dim))
        # self.register_buffer("w_loop", nn.Parameter(input_dim, input_dim))

        # layer-specific self-loop relation parameter
        self.loop_relation = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.loop_relation)

        self.use_attention = use_attention
        if self.use_attention:
            self.num_heads = num_heads
            self.query = nn.Parameter(torch.zeros(num_heads, input_dim * 2 // num_heads))
            nn.init.kaiming_uniform_(self.query, 0.2, mode="fan_in")
            self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=0.2)


    def forward(self, graph, input):
        """
        CompGCN forward pass is the average of direct, inverse, and self-loop messages
        """

        # out graph -> the original graph without inverse edges
        edge_list = graph.edge_list
        out_edges = edge_list[edge_list[:, 2] < graph.num_relation // 2]
        out_edge_weight = torch.ones(len(out_edges), device=self.device)
        out_graph = type(graph)(out_edges, edge_weight=out_edge_weight, num_node=graph.num_node,
                            num_relation=graph.num_relation, meta_dict=graph.meta_dict, **graph.data_dict)
        out_graph.relation_embs = graph.relation_embs
        out_graph.dir_transform = self.w_out

        # in graph -> the graph with only inverse edges
        in_edges = edge_list[edge_list[:, 2] >= graph.num_relation // 2]
        in_edge_weight = torch.ones(len(in_edges), device=self.device)
        in_graph = type(graph)(in_edges, edge_weight=in_edge_weight, num_node=graph.num_node,
                                num_relation=graph.num_relation, meta_dict=graph.meta_dict, **graph.data_dict)
        in_graph.relation_embs = graph.relation_embs
        in_graph.dir_transform = self.w_in

        # self_loop graph -> the graph with only self-loop relation type
        node_in = node_out = torch.arange(graph.num_node, device=self.device)
        loop = torch.stack([node_in, node_out], dim=-1)
        loop_relation = torch.zeros(len(loop), 1, dtype=torch.long, device=self.device)
        loop_edges = torch.cat([loop, loop_relation], dim=-1)
        self_edge_weight = torch.ones(graph.num_node, device=self.device)
        self_loop_graph = type(graph)(loop_edges, edge_weight=self_edge_weight, num_node=graph.num_node,
                                num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        self_loop_graph.relation_embs = self.loop_relation
        self_loop_graph.dir_transform = self.w_loop

        # message passing
        out_update = self.plain_forward(out_graph, input)
        in_update = self.plain_forward(in_graph, input)
        loop_update = self.plain_forward(self_loop_graph, input)
        update = (out_update + in_update + loop_update) / 3.0
        output = self.combine(input, update)
        return output, self.relation_update(graph.relation_embs)

    def plain_forward(self, graph, input):
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input)
        return update

    def message(self, graph, input):
        node_in, node_out, relation = graph.edge_list.t()

        relation_input = graph.relation_embs
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # direction-wise message transformation
        message = torch.mm(message, graph.dir_transform)

        if self.use_attention:
            key = torch.stack([message, input[node_out]], dim=-1)
            key = key.view(-1, *self.query.shape)
            weight = torch.einsum("hd, nhd -> nh", self.query, key)
            weight = self.leaky_relu(weight)
            weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
            attention = weight.exp()
            normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
            attention = attention / (normalizer + self.eps)
            value = message.view(-1, self.num_heads, self.query.shape[-1] // 2)
            attention = attention.unsqueeze(-1).expand_as(value)
            message = (attention * value).flatten(1)

        return message

    def aggregate(self, graph, message):

        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        # node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        # edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        # edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(TrueCompGCNConv, self).message_and_aggregate(graph, input)
        # print("doing rspmm")
        input = input.flatten(1)

        degree_out = graph.degree_out.unsqueeze(-1) + 1

        relation_input = graph.relation_embs.to(input.device)
        adjacency = graph.adjacency.transpose(0, 1)
        dir_weight = graph.dir_transform

        # TODO include message transformation by dir_weight into the result of rspmm
        # TODO we don't need boundary in this compgcn version

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            #update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            #update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            #update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum) / degree_out
            sq_mean = (sq_sum) / degree_out
            #max = torch.max(max, boundary)
            #min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = torch.mm(update, dir_weight)
        return update.view(len(update), -1)

    def combine(self, input, update):
        # in CompGCN, we just return updated states, no aggregation with inputs
        # update = update
        # output = self.linear(torch.cat([input, update], dim=-1))
        output = update if self.aggregate_func != "pna" else self.linear(update)
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output