import numpy
import torch
import pickle

from torch import nn
from torchdrug import utils
from torchdrug.data import Graph
from torch_sparse import spmm

from typing import Dict, List


def page_rank(edge_index, num_nodes=None, max_iter=1000, alpha=0.05, epsilon=1e-4, x0=None):
    # prepare adjacency
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    node_in, node_out = edge_index
    node_in, node_out = torch.cat([node_in, node_out]), torch.cat([node_out, node_in])
    degree = node_in.bincount(minlength=num_nodes)
    value = torch.ones(len(node_in), device=edge_index.device)
    value = value / degree
    adj = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), value, (num_nodes, num_nodes))

    # prepare x0
    if x0 is None:
        x0 = torch.full(size=(num_nodes,), fill_value=1.0 / num_nodes)

    # power iteration
    device = torch.device("cuda")
    adj = adj.to(device)
    x0 = x0.to(device)
    no_batch = x0.ndim < 2
    if no_batch:
        x0 = x0.unsqueeze(dim=-1)
    # power iteration
    x_old = x = x0
    beta = 1.0 - alpha
    for i in range(max_iter):
        x = torch.sparse.addmm(
            x0, adj, x, beta=alpha, alpha=beta,
        )
        x = torch.nn.functional.normalize(x, dim=0, p=1)
        diff = torch.linalg.norm(x - x_old, ord=float("+inf"), axis=0)
        mask = diff > epsilon
        if not mask.any():
            print(f"Converged after {i} iterations up to {epsilon}.")
            break
        x_old = x
    else:
        print(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")
    if no_batch:
        x = x.squeeze(dim=-1)

    if x.ndim < 2:
        return x
    return x.t()


class NodePieceTokenizer:

    def __init__(self,
                 num_anchors: int,
                 ancs_per_node: int,
                 train_graph: Graph,
                 valid_graph: Graph,
                 test_graph: Graph,
                 strategy: Dict[str, float] = {"degree": 0.5, "random": 0.5},
                 sp_limit: int = 25,
                 bfs_max_hops: int = 5,
                 wikikg: bool = None,
                 ):

        self.num_anchors = num_anchors
        self.ancs_per_node = ancs_per_node
        self.sp_limit = sp_limit
        self.seed = 42  # TODO
        self.strategy = {"degree": 0.0, "pagerank": 0.0, "random": 0.0}
        self.strategy.update(strategy)
        self.bfs_max_hops = bfs_max_hops

        self.PADDING_TOKEN = self.num_anchors

        # wikikg2_nodepiece.yaml is a path to the vocab or False
        if wikikg is None:
            self.anchors, self.hashes = self.tokenize(train_graph, valid_graph, test_graph)
            self.token2id = {anchor: i for i, anchor in enumerate(self.anchors + [self.PADDING_TOKEN])}
        else:
            self.anchors, self.hashes = self.load_wikikg_vocab(wikikg)

    def tokenize(self, train_graph: Graph, valid_graph: Graph, test_graph: Graph):

        # create a filename
        fname = f"td_{self.num_anchors}_ancs"
        fname += f"_d{self.strategy['degree']}_p{self.strategy['pagerank']}_r{self.strategy['random']}"
        fname += f"_{self.sp_limit}sp"
        fname += ".pkl"

        anchors = self.find_anchors(train_graph)
        hashes = [self.anchor_search(graph, anchors, self.bfs_max_hops) for graph in (train_graph, valid_graph, test_graph)]

        return anchors, hashes

    def find_anchors(self, graph: Graph):

        # mine anchors in the target graph according to the strategy
        ratios = list(self.strategy.values())
        strategies = list(self.strategy.keys())

        # map a certain number of anchors to each strategy
        num_anchors = get_split(total=self.num_anchors, ratios=ratios)
        num_nodes = graph.edge_list[:, :2].unique().shape[0]

        anchors = []
        for strategy, num_ancs in zip(strategies, num_anchors):
            print(f"Computing {strategy} anchors")
            if strategy == "degree" and num_ancs > 0:
                unique, counts = torch.unique(graph.edge_list[:, :2], return_counts=True)
                ids = torch.argsort(counts, descending=True)
                ancs = ids.detach().cpu().numpy()
            elif strategy == "pagerank" and num_ancs > 0:
                ancs = torch.argsort(page_rank(edge_index=graph.edge_list.t()[:2], num_nodes=num_nodes), descending=True).detach().cpu().numpy()
            elif strategy == "random" and num_ancs > 0:
                generator = numpy.random.default_rng(self.seed)
                ancs = generator.permutation(graph.edge_list[:, :2].max().item())
            else:
                continue

            # TODO torch.isin() appears only from Torch 1.10, so for now process via numpy
            unique_anchors = ancs[~numpy.isin(ancs, anchors)]
            unique_anchors = unique_anchors[:num_ancs]

            anchors.extend(unique_anchors.tolist())

        return anchors

    def anchor_search(self, graph: Graph, anchors: List[int], max_iter: int = 5):

        # we'll use BFS similar to the procedure in PyKEEN
        # https://github.com/pykeen/pykeen/blob/e49a3551f4a54ea246a5233ce6d0aa0291f1c313/src/pykeen/nn/node_piece/anchor_search.py#L134
        num_entities = graph.num_node.item()
        graph = graph.undirected(add_inverse=True)
        edge_list = graph.edge_list[:, :2].t()

        num_unique_nodes = edge_list.unique(return_counts=False).shape[0]
        self_loops = torch.arange(num_entities, device=graph.device).repeat(2, 1)
        edge_list = torch.cat([edge_list, self_loops], dim=-1)
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        values = torch.ones_like(edge_weight).bool()
        # adjacency = torch.sparse_coo_tensor(indices=edge_list, values=values, size=(num_entities, num_entities), dtype=torch.bool)
        # adjacency = torchdrug.utils.sparse_coo_tensor(edge_list, edge_weight, (graph.num_node, graph.num_node))

        num_anchors = len(anchors)
        anchors = torch.tensor(anchors, dtype=torch.long, device=graph.device)

        # an array storing whether node i is reachable by anchor j
        reachable = torch.zeros((num_entities, num_anchors), dtype=torch.bool, device=graph.device)
        reachable[anchors] = torch.eye(num_anchors, dtype=torch.bool, device=graph.device)

        # an array indicating whether a node is closed, i.e., has found at least $k$ anchors
        final = torch.zeros((num_entities,), dtype=torch.bool, device=graph.device)

        # the output
        pool = torch.empty((num_entities, num_anchors), dtype=torch.int8, device=graph.device).fill_(127)
        pool[anchors, torch.arange(len(anchors), dtype=torch.long, device=graph.device)] = 0
        k = self.sp_limit

        # TODO: take all (q-1) hop neighbors before selecting from q-hop
        old_reachable = reachable
        for i in range(max_iter):
            # propagate one hop
            # TODO workaround until the fix in torch_sparse
            reachable = spmm(index=edge_list, value=values.float(), m=num_entities, n=num_entities, matrix=reachable.float()) > 0.0

            # convergence check
            if (reachable == old_reachable).all():
                print(f"Search converged after iteration {i} without all nodes being reachable.")
                break
            newly_reached = reachable ^ old_reachable
            old_reachable = reachable
            # copy pool if we have seen enough anchors and have not yet stopped
            num_reachable = reachable.sum(axis=1)
            enough = num_reachable >= k
            mask = enough & ~final
            print(f"Iteration {i}: {enough.sum()} / {num_unique_nodes} closed nodes.")
            # pool[mask] = (reachable[mask] * (i+1)).char()
            pool[newly_reached] = i + 1
            # stop once we have enough
            final |= enough
            if final.all():
                break

        values, indices = torch.sort(pool, dim=-1)
        # max value 127 to padding token
        indices[values == 127] = self.PADDING_TOKEN
        hashes = indices[:, :self.ancs_per_node]
        return hashes.detach().cpu()


    def load_wikikg_vocab(self, path):
        # load and process the pre-trained vocabulary for the custom split of WikiKG 2
        anchors, vocab = pickle.load(open(path, "rb"))
        token2id = {t: i for i, t in enumerate(anchors)}

        # by default, anchors contain K anchors + 5 auxiliary tokens
        anchors = torch.tensor(anchors, dtype=torch.long)[:-5]

        hashes = [
            [token2id[token] for token in
             vals['ancs'][:min(self.ancs_per_node, len(vals['ancs']))]] + [self.PADDING_TOKEN] * (self.ancs_per_node - len(vals['ancs']))
            for entity, vals in vocab.items()
        ]

        hashes = torch.tensor(hashes, dtype=torch.long)
        final_hashes = [hashes, hashes, hashes]
        return anchors, final_hashes


def get_split(total: int, ratios: List[float]):

    cum_ratio = numpy.cumsum(ratios)
    cum_ratio[-1] = 1.0
    cum_ratio = numpy.r_[numpy.zeros(1), cum_ratio]
    split_points = (cum_ratio * total).astype(numpy.int64)
    sizes = numpy.diff(split_points)

    return tuple(sizes)
