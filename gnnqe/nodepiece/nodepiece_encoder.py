import torch
import numpy as np

from torch import nn
from collections import defaultdict
from torchdrug import utils
from torchdrug.layers import functional
from tqdm import tqdm

from .compgcn_encoder import CompGCN
from .nodepiece_tokenizer import NodePieceTokenizer

import random


class NodePieceEncoder(nn.Module):

    def __init__(self, num_relations, use_inverses, embedding_dim, pooler, train_graph, valid_graph, test_graph, num_anchors, ancs_per_node,
                 rel_context, gnn, gnn_dims=None, message_func="distmult", aggregate_func="pna", short_cut=False,
                 concat_hidden=False, encoder_drop=0.1, activation="relu", layer_norm=False, use_boundary=False,
                 use_attention=False, num_heads=4, wikikg=None):

        super().__init__()
        # no need for the anchor tokenizer in the fully inductive LP setup

        self.num_anchors = num_anchors
        self.ancs_per_node = ancs_per_node
        self.rel_context = rel_context
        self.use_gnn = gnn
        self.pooler = pooler
        self.subbatch = 2000
        self.use_inverses = use_inverses

        # relation_tokens, last one is PADDING
        # TODO: initialization based on the scoring function
        self.relations = nn.Parameter(torch.empty(num_relations + 1, embedding_dim))
        self.padding_idx = num_relations
        self.embedding_dim = embedding_dim

        if self.num_anchors > 0:
            self.tokenizer = NodePieceTokenizer(
                num_anchors=num_anchors,
                ancs_per_node=ancs_per_node,
                train_graph=train_graph,
                valid_graph=valid_graph,
                test_graph=test_graph,
                wikikg=wikikg
            )
        else:
            self.tokenizer = None

        # pooler function
        if self.pooler == "cat":
            input_tokens = rel_context if num_anchors == 0 else rel_context + ancs_per_node
            self.set_enc = nn.Sequential(
                nn.Linear(embedding_dim * input_tokens, embedding_dim * 2), nn.Dropout(encoder_drop), nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
        elif self.pooler == "set":
            self.set_enc = DeepSet(hidden_dim=embedding_dim)
        elif self.pooler == "2sum":
            # not learnable random projections
            self.set_enc = [
                nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                              nn.Linear(embedding_dim, embedding_dim)),
                nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                              nn.Linear(embedding_dim, embedding_dim))
            ]

            self.dec = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                                     nn.Linear(embedding_dim, embedding_dim))

        if self.use_gnn:
            self.gnn_encoder = CompGCN(input_dim=self.embedding_dim,
                                       hidden_dims=gnn_dims,
                                       edge_input_dim=self.embedding_dim,
                                       message_func=message_func,
                                       aggregate_func=aggregate_func,
                                       short_cut=short_cut,
                                       concat_hidden=concat_hidden,
                                       activation=activation,
                                       layer_norm=layer_norm,
                                       use_boundary=use_boundary,
                                       use_attention=use_attention,
                                       num_heads=num_heads)

        # tokenize train graph, add inverses if necessary
        # if self.use_inverses:
        #     train_graph = train_graph.undirected(add_inverse=True)
        train_edges = train_graph.edge_list
        if self.use_inverses:
            train_edges_inv = train_edges[:, [1, 0, 2]]
            train_edges_inv[:, 2] += num_relations // 2
        else:
            train_edges_inv = torch.tensor([], device=train_edges.device, dtype=torch.long)
        train_outgoing_rels, train_incoming_rels = defaultdict(set), defaultdict(set)
        for h, t, r in tqdm(torch.cat([train_edges, train_edges_inv], dim=0).tolist()):
            train_outgoing_rels[h].add(r)
            train_incoming_rels[t].add(r)

        # tokenize valid graph, add inverses if necessary
        # valid_edges = valid_graph.edge_list
        mask = valid_graph.match(train_edges)[0]
        edge_mask = ~functional.as_mask(mask, valid_graph.num_edge)
        valid_edges = valid_graph.edge_mask(edge_mask).edge_list
        if self.use_inverses:
            valid_edges_inv = valid_edges[:, [1, 0, 2]]
            valid_edges_inv[:, 2] += num_relations // 2
        else:
            valid_edges_inv = torch.tensor([], device=train_edges.device, dtype=torch.long)
        valid_outgoing_rels, valid_incoming_rels = defaultdict(set), defaultdict(set)
        for h, t, r in tqdm(torch.cat([valid_edges, valid_edges_inv], dim=0).tolist()):
            valid_outgoing_rels[h].add(r)
            valid_incoming_rels[t].add(r)

        # tokenize test graph
        # test_edges = test_graph.edge_list
        mask = test_graph.match(train_edges)[0]
        edge_mask = ~functional.as_mask(mask, test_graph.num_edge)
        test_edges = test_graph.edge_mask(edge_mask).edge_list
        if self.use_inverses:
            test_edges_inv = test_edges[:, [1, 0, 2]]
            test_edges_inv[:, 2] += num_relations // 2
        else:
            test_edges_inv = torch.tensor([], device=train_edges.device, dtype=torch.long)
        test_outgoing_rels, test_incoming_rels = defaultdict(set), defaultdict(set)
        for h, t, r in tqdm(torch.cat([test_edges, test_edges_inv], dim=0).tolist()):
            test_outgoing_rels[h].add(r)
            test_incoming_rels[t].add(r)

        len_stats = [len(v) for k, v in train_outgoing_rels.items()]
        print(f"Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(len_stats, 66)}, max: {max(len_stats)} ")

        # Now we need to create the final dict of relations:
        # train nodes -> built from the training graph
        # unique val nodes -> built from the valid graph
        # unique test nodes -> built from the test graph
        train_nodes = set(torch.unique(train_graph.edge_list[:, [0, 1]]).numpy().tolist())
        val_nodes = set(torch.unique(valid_graph.edge_list[:, [0, 1]]).numpy().tolist())
        test_nodes = set(torch.unique(test_graph.edge_list[:, [0, 1]]).numpy().tolist())
        unique_val_nodes = sorted(val_nodes.difference(train_nodes))
        unique_test_nodes = sorted(test_nodes.difference(train_nodes))

        final_out_dict, final_in_dict = self.build_final_dict(
            (train_outgoing_rels, train_incoming_rels),
            (valid_outgoing_rels, valid_incoming_rels),
            (test_outgoing_rels, test_incoming_rels),
            train_nodes, unique_val_nodes, unique_test_nodes
        )

        # build a vocab of anchors
        if self.tokenizer is not None:
            anchor_hashes = self.tokenizer.hashes[0]
            t_unique_val_nodes = torch.tensor(unique_val_nodes, dtype=torch.long)
            t_unique_test_nodes = torch.tensor(unique_test_nodes, dtype=torch.long)
            anchor_hashes[t_unique_val_nodes] = self.tokenizer.hashes[1][t_unique_val_nodes]
            anchor_hashes[t_unique_test_nodes] = self.tokenizer.hashes[2][t_unique_test_nodes]
            unique_anchor_hashes = torch.unique(anchor_hashes, dim=0).shape[0]
            unique_anchor_ratio = unique_anchor_hashes / anchor_hashes.shape[0]
            print(f"Unique anchor hashes: {unique_anchor_hashes} / {len(anchor_hashes)}, ratio: {unique_anchor_ratio}")

            self.register_buffer("anchor_hashes", anchor_hashes)
            self.anchor_embeddings = nn.Embedding(num_anchors + 1, embedding_dim)

        # build a vocab of relations
        self.out_context, self.in_context = self.sample_relational_context(
            outgoing_dict=final_out_dict,
            incoming_dict=final_in_dict if self.pooler == "2sum" else None
        )
        unique_hashes = torch.unique(self.out_context, dim=0).shape[0]
        unique_ratio = unique_hashes / self.out_context.shape[0]
        print(f"Unique hashes in the rel context: {unique_hashes} / {len(self.out_context)}, ratio {unique_ratio}")
        if self.num_anchors > 0:
            total_unique = torch.unique(torch.cat([anchor_hashes, self.out_context], dim=-1), dim=0).shape[0]
            print(f"Total Unique hashes: {total_unique} / {len(self.out_context)}, ratio {total_unique / len(self.out_context)}")

        self.register_buffer("node_hashes", self.out_context)

        if self.pooler == "2sum":
            self.register_buffer("node_inc_hashes", self.in_context)

        self.cache = None
        self.entity_cache = None


    def sample_relational_context(self, outgoing_dict, incoming_dict=None):
        # sample relational context from the given dictionaries
        # if the incoming dict is not None - tokenize incoming relations as well
        out_relational_context = torch.tensor([
            random.sample(outgoing_dict[i], k=min(self.rel_context, len(outgoing_dict[i]))) + [self.padding_idx] * (
                self.rel_context - min(self.rel_context, len(outgoing_dict[i]))
            )
            for i in range(len(outgoing_dict))
        ], dtype=torch.long)

        if incoming_dict:
            in_relational_context = torch.tensor([
                random.sample(incoming_dict[i], k=min(self.rel_context, len(incoming_dict[i]))) + [self.padding_idx] * (
                        self.rel_context - min(self.rel_context, len(incoming_dict[i]))
                )
                for i in range(len(incoming_dict))
            ], dtype=torch.long)
        else:
            in_relational_context = None

        return out_relational_context if not incoming_dict else out_relational_context, in_relational_context

    def build_final_dict(self, train_context, val_context, test_context, train_nodes, val_nodes, test_nodes):

        # final_out_dict = copy.deepcopy(train_context[0])
        # final_in_dict = copy.deepcopy(train_context[1])

        # use get() to cater for the case when a node might not have any relations (weird but who knows)
        final_out_dict = {k: train_context[0].get(k, []) for k in train_nodes}
        final_in_dict = {k: train_context[1].get(k, []) for k in train_nodes}

        val_out_dict = {k: val_context[0].get(k, []) for k in val_nodes}
        val_in_dict = {k: val_context[1].get(k, []) for k in val_nodes}

        test_out_dict = {k: test_context[0].get(k, []) for k in test_nodes}
        test_in_dict = {k: test_context[1].get(k, []) for k in test_nodes}

        final_out_dict.update(val_out_dict)
        final_out_dict.update(test_out_dict)

        final_in_dict.update(val_in_dict)
        final_in_dict.update(test_in_dict)

        return final_out_dict, final_in_dict

    def init_relations(self, score_func, max_score):
        if score_func == "transe":
            nn.init.uniform_(self.relations, -max_score / self.embedding_dim, max_score / self.embedding_dim)
        elif score_func == "rotate":
            nn.init.uniform_(self.relations, -max_score * 2 / self.embedding_dim, max_score * 2 / self.embedding_dim)
            pi = torch.acos(torch.zeros(1)).item() * 2
            self.relation_scale = pi * self.embedding_dim / max_score / 2
            self.relations *= self.relation_scale
        elif score_func == "complex":
            nn.init.uniform_(self.relations, -0.5, 0.5)
        elif score_func == "distmult":
            nn.init.uniform_(self.relations, -0.5, 0.5)


    def pool_hashes(self, node_embs):

        if self.pooler == "cat":
            node_embs = node_embs.view(node_embs.shape[0], -1)
            pooled = self.set_enc(node_embs)
        elif self.pooler == "set":
            pooled = self.set_enc(node_embs)
        elif self.pooler == "2sum":
            node_embs = node_embs.sum(-2)  # (bs, 2, dim)
            pooled = torch.stack([self.set_enc[i].to(node_embs.device)(node_embs[:, i, :]) for i in range(2)], dim=1)
            pooled = pooled.sum(dim=1)
            pooled = self.dec(pooled)

        return pooled

    def encode_by_index(self, idx, mode):

        if len(idx.shape) > 1:
            idx, rets = torch.unique(idx, return_inverse=True)
        else:
            rets = None

        rel_hashes = self.node_hashes

        node_hashes = rel_hashes[idx]  # (bs, rel_context)
        node_embs = self.relations[node_hashes]  # (bs, rel_context, emb_dim)

        if self.num_anchors > 0:
            anchor_hashes = self.anchor_hashes[idx]  # (bs, ancs_per_node)
            anchor_embs = self.anchor_embeddings(anchor_hashes)  # (bs, ancs_per_node, emb_dim)
            node_embs = torch.cat([anchor_embs, node_embs], dim=-2)  # (bs, ancs_per_node + rel_context, emb_dim)

        if self.pooler == "2sum":
            #inc_rel_hashes = self.train_inc_hashes if mode != "test" else self.test_inc_hashes
            inc_rel_hashes = self.node_inc_hashes
            inc_node_hashes = inc_rel_hashes[idx]
            inc_node_embs = self.relations[inc_node_hashes]

            if self.num_anchors == 0:
                node_embs = torch.stack([node_embs, inc_node_embs], dim=1)  # (bs, 2, rel_context, emb_dim)
            else:
                # output shape: (bs, 2, ancs_per_node + rel_context, emb_dim)
                node_embs = torch.stack([node_embs, torch.cat([anchor_embs, inc_node_embs], dim=-2)], dim=1)

        embs = self.pool_hashes(node_embs)  # (bs, emb_dim)

        return embs if rets is None else embs[rets]

    def get_all_representations(self, mode):

        if not self.training:  # we only cache embeddings in the eval mode
            if self.entity_cache is not None:
                return self.entity_cache
        else:
            self.entity_cache = None

        num_ents = self.node_hashes.shape[0]
        output = torch.empty((num_ents, self.embedding_dim), dtype=torch.float, device=self.node_hashes.device)

        for i in torch.arange(num_ents, device=self.node_hashes.device).split(self.subbatch):
            embs = self.encode_by_index(i, mode)
            output[i, :] = embs

        if not self.training:
            self.entity_cache = output

        return output

    def encode_gnn(self, graph, mode):
        if not self.training and mode == "valid":  # we only cache embeddings in the eval mode
            if self.cache is not None:
                return self.cache[0], self.cache[1]
        else:
            self.cache = None

        graph = graph.undirected(add_inverse=True)
        node_features = self.get_all_representations(mode)
        relations = self.relations[:-1]  # all relations except the PADDING
        node_features, relations = self.gnn_encoder(graph, node_features, relations)

        if not self.training and mode == "valid":
            self.cache = [node_features, relations]

        return node_features, relations

    def get_entities_and_relations(self, graph, mode):

        if self.use_gnn:
            return self.encode_gnn(graph, mode)
        else:
            return self, self.relations

    def materialize_embeddings(self, graph, mode):
        if self.use_gnn:
            return self.encode_gnn(graph, mode)
        else:
            return self.get_all_representations(mode), self.relations


    def __getitem__(self, item):
        mode = self.mode
        return self.encode_by_index(item, mode)




class DeepSet(torch.nn.Module):

    def __init__(self, hidden_dim=64):

        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, dim=-2):
        x = self.encoder(x).mean(dim)
        x = self.decoder(x)

        return x





