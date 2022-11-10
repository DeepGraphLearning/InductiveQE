from collections.abc import Sequence

import torch
from torch import nn

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.layers.functional import embedding
embedding.backend = "native" #if not torch.cuda.is_available() else "fast"
from torchdrug.core import Registry as R

from .nodepiece_encoder import NodePieceEncoder


@R.register("model.NodePiece")
class NodePiece(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, use_inverses=False, num_anchors=0, ancs_per_node=0, rel_context=0,
                 train_graph=None, valid_graph=None, test_graph=None, gnn=False, layer_norm=False, pooler="cat",
                 scoring_function="distmult", message_func="distmult", aggregate_func="pna", short_cut=False,
                 activation="relu", concat_hidden=False, remove_one_hop=False, use_boundary=False,
                 use_attention=False, num_heads=4, wikikg=None):
        super(NodePiece, self).__init__()


        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        num_relation = int(num_relation)
        double_relation = num_relation * 2 if use_inverses else num_relation
        #self.gnn_dims = list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.message_func = message_func

        self.encoder = NodePieceEncoder(
            num_relations=double_relation,
            use_inverses=use_inverses,
            embedding_dim=input_dim,
            pooler=pooler,
            train_graph=train_graph,
            valid_graph=valid_graph,
            test_graph=test_graph,
            num_anchors=num_anchors,
            ancs_per_node=ancs_per_node,
            rel_context=rel_context,
            gnn=gnn,
            gnn_dims=hidden_dims,
            message_func=message_func,
            aggregate_func=aggregate_func,
            short_cut=self.short_cut,
            concat_hidden=self.concat_hidden,
            activation=activation,
            layer_norm=layer_norm,
            use_boundary=use_boundary,
            use_attention=use_attention,
            num_heads=num_heads,
            wikikg=wikikg,
        )

        if scoring_function == "distmult":
            self.score_function = functional.distmult_score
        elif scoring_function == "transe":
            self.score_function = functional.transe_score
        elif scoring_function == "rotate":
            self.score_function = functional.rotate_score
        elif scoring_function == "complex":
            self.score_function = functional.complex_score

        self.encoder.init_relations(scoring_function, max_score=12)  # 12 is default for tdrug TransE and RotatE

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None, mode=None):
        # if all_loss is not None:
        #     graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape

        # graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        self.encoder.mode = mode

        entities, relations = self.encoder.get_entities_and_relations(graph, mode)

        score = self.score_function(entities, relations, h_index, t_index, r_index)

        return score.view(shape)

