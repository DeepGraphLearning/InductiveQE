import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import Query


class LogicalQueryDataset(data.KnowledgeGraphDataset):
    """Logical query dataset."""

    struct2type = {
        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r")): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
    }

    def load_pickle(self, path, query_types=None, union_type="DNF", verbose=0):
        """
        Load the dataset from pickle dumps (BetaE format).

        Parameters:
            path (str): path to pickle dumps
            query_types (list of str, optional): query types to load.
                By default, load all query types.
            union_type (str, optional): which union type to use, ``DNF`` or ``DM``
            verbose (int, optional): output verbose level
        """
        query_types = query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, union_type)
                elif query_type[query_type.find("-") + 1:] != union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}

        with open(os.path.join(path, "id2ent.pkl"), "rb") as fin:
            entity_vocab = pickle.load(fin)
        with open(os.path.join(path, "id2rel.pkl"), "rb") as fin:
            relation_vocab = pickle.load(fin)
        triplets = []
        num_samples = []
        for split in ["train", "valid", "test"]:
            triplet_file = os.path.join(path, "%s.txt" % split)
            with open(triplet_file) as fin:
                if verbose:
                    fin = tqdm(fin, "Loading %s" % triplet_file, utils.get_line_count(triplet_file))
                num_sample = 0
                for line in fin:
                    h, r, t = [int(x) for x in line.split()]
                    triplets.append((h, t, r))
                    num_sample += 1
                num_samples.append(num_sample)
        self.load_triplet(triplets, entity_vocab=entity_vocab, relation_vocab=relation_vocab)
        fact_mask = torch.arange(num_samples[0])
        # self.graph is the full graph without missing edges
        # self.fact_graph is the training graph
        self.fact_graph = self.graph.edge_mask(fact_mask)

        queries = []
        types = []
        easy_answers = []
        hard_answers = []
        num_samples = []
        max_query_length = 0

        for split in ["train", "valid", "test"]:
            if verbose:
                pbar = tqdm(desc="Loading %s-*.pkl" % split, total=3)
            with open(os.path.join(path, "%s-queries.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            if verbose:
                pbar.update(1)
            type2queries = {self.struct2type[k]: v for k, v in struct2queries.items()}
            type2queries = {k: v for k, v in type2queries.items() if k in self.type2id}
            if split == "train":
                with open(os.path.join(path, "%s-answers.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                query2hard_answers = defaultdict(set)
                if verbose:
                    pbar.update(2)
            else:
                with open(os.path.join(path, "%s-easy-answers.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)
                with open(os.path.join(path, "%s-hard-answers.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)

            num_sample = sum([len(q) for t, q in type2queries.items()])
            if verbose:
                pbar = tqdm(desc="Processing %s queries" % split, total=num_sample)
            for type in type2queries:
                struct_queries = sorted(type2queries[type])
                for query in struct_queries:
                    easy_answers.append(query2easy_answers[query])
                    hard_answers.append(query2hard_answers[query])
                    query = Query.from_nested(query)
                    queries.append(query)
                    max_query_length = max(max_query_length, len(query))
                    types.append(self.type2id[type])
                    if verbose:
                        pbar.update(1)
            num_samples.append(num_sample)

        self.queries = queries
        self.types = types
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        self.num_samples = num_samples
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": functional.as_mask(easy_answer, self.num_entity),
            "hard_answer": functional.as_mask(hard_answer, self.num_entity),
        }

    def __len__(self):
        return len(self.queries)

    def __repr__(self):
        lines = [
            "#entity: %d" % self.num_entity,
            "#relation: %d" % self.num_relation,
            "#triplet: %d" % self.num_triplet,
            "#query: %d" % len(self.queries),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("dataset.FB15kLogicalQuery")
class FB15kLogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "FB15k-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)


@R.register("dataset.FB15k237LogicalQuery")
class FB15k237LogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "FB15k-237-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)


@R.register("dataset.NELL995LogicalQuery")
class NELL995LogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "NELL-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)



@R.register("dataset.InductiveFB15k237Comp")
class InductiveFB15k237Comp(LogicalQueryDataset):

    url = "https://zenodo.org/record/7306046/files/%s.zip"

    md5 = {
        550: "e78bb9a7de9bd55813bb17f57941303c",
        300: "4db5c172acf83f676c9cf6589e033d7e",
        217: "9fde4563c619dc4d2b81af200cf7bc6b",
        175: "29ee1dbed7662740a2f001a0c6df8911",
        150: "61b545de8e5cdb04832f27842d8c0175",
        134: "cd8028c9674dc81f38cd17b03af43fe1",
        122: "272d2cc1e3f98f76d02daaf066f9d653",
        113: "e4ea60448e918c62779cfa757a096aa9",
        106: "6f9a1dcf22108074fb94a05b8377a173",
        "wikikg": "fa30b189436ab46a2ff083dd6a5e6e0b"
    }

    def __init__(self, path, ratio, inverse_projection=False, query_types=None, union_type="DNF", verbose=1,
                 train_patterns=('1p', '2p', '3p', '2i', '3i', '2in', '3in', 'inp', 'pni', 'pin')):
        # Data Loader for inductive splits, ratio specifies the ratio "# inference entities / # training entities"
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        # # Download data if it's not there
        zip_file = utils.download(self.url % str(ratio), path, md5=self.md5[ratio])
        path = os.path.join(path, str(ratio))
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.inverse_projection = inverse_projection
        self.verbose = verbose
        query_types = query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, union_type)
                elif query_type[query_type.find("-") + 1:] != union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}
        self.union_type = union_type

        # Space of entities 0 ... N is split into 3 sets
        # Train node IDs: 0 ... K
        # Val inference ids: K ... K+M
        # Test inference ids: K+M .... N
        try:
            train_triplets = self.load_file(os.path.join(path, "train_graph.txt"))
            val_inference = self.load_file(os.path.join(path, "val_inference.txt"))
            test_inference = self.load_file(os.path.join(path, "test_inference.txt"))
        except FileNotFoundError:
            print("Loading .pt files")
            train_triplets = self.load_pt(os.path.join(path, "train_graph.pt"))
            val_inference = self.load_pt(os.path.join(path, "val_inference.pt"))
            test_inference = self.load_pt(os.path.join(path, "test_inference.pt"))

        entity_vocab, relation_vocab, tr_nodes, vl_nodes, ts_nodes = self.build_vocab(train_triplets, val_inference, test_inference)
        entity_vocab, inv_entity_vocab = self._standarize_vocab(entity_vocab, None)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, None)

        num_node = len(entity_vocab) if entity_vocab else None
        num_relation = len(relation_vocab) if relation_vocab else None

        # Training graph: only training triples
        self.train_graph = data.Graph(train_triplets, num_node=len(tr_nodes), num_relation=num_relation)

        # Validation graph: training triples (0..K) + new validation inference triples (K+1...K+M)
        self.valid_graph = data.Graph(train_triplets + val_inference, num_node=num_node, num_relation=num_relation)

        # Test graph: training triples (0..K) + new test inference triples (K+M+1... N)
        self.test_graph = data.Graph(train_triplets + test_inference, num_node=num_node, num_relation=num_relation)

        # Full graph (aux purposes)
        self.graph = data.Graph(train_triplets + val_inference + test_inference,
                                num_node=num_node, num_relation=num_relation)
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_entity_vocab = inv_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

        # Need those for evaluation
        self.valid_nodes = torch.tensor(vl_nodes, dtype=torch.long)
        self.test_nodes = torch.tensor(ts_nodes, dtype=torch.long)

        queries = []
        type_ids = []
        easy_answers = []
        hard_answers = []
        num_samples = []
        num_entity_for_sample = []
        max_query_length = 0

        type2struct = {v: k for k, v in self.struct2type.items()}
        filtered_training_structs = tuple([type2struct[x] for x in train_patterns])
        for split in ["train", "valid", "test"]:
            with open(os.path.join(path, "%s_queries.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            if split == "train":
                query2hard_answers = defaultdict(lambda: defaultdict(set))
                with open(os.path.join(path, "%s_answers_hard.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
            else:
                with open(os.path.join(path, "%s_answers_easy.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                with open(os.path.join(path, "%s_answers_hard.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
            num_sample = 0
            structs = sorted(struct2queries.keys(), key=lambda s: self.struct2type[s])
            if verbose:
                structs = tqdm(structs, "Loading %s queries" % split)
            for struct in structs:
                query_type = self.struct2type[struct]
                if query_type not in self.type2id:
                    continue
                # filter complex patterns ip, pi, 2u, up from training queries - those will be eval only
                if split == "train" and struct not in filtered_training_structs:
                    print(f"Skipping {query_type} - this will be used in evaluation")
                    continue
                struct_queries = sorted(struct2queries[struct])
                for query in struct_queries:
                    # The dataset format is slightly different from BetaE's
                    easy_answers.append(query2easy_answers[struct][query])
                    hard_answers.append(query2hard_answers[struct][query])
                    query = Query.from_nested(query)
                    #query = self.to_postfix_notation(query)
                    max_query_length = max(max_query_length, len(query))
                    queries.append(query)
                    type_ids.append(self.type2id[query_type])
                num_sample += len(struct_queries)
            num_entity_for_sample += [getattr(self, "%s_graph" % split).num_node.item()] * num_sample
            num_samples.append(num_sample)

        self.queries = queries
        self.types = type_ids
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        self.num_samples = num_samples
        self.num_entity_for_sample = num_entity_for_sample
        self.max_query_length = max_query_length

    def load_file(self, path):
        triplets = []
        with open(path) as fin:
            for line in fin:
                h, r, t = [int(x) for x in line.split()]
                triplets.append((h, t, r))

        return triplets

    def load_pt(self, path):
        triplets = torch.load(path, map_location="cpu")
        return triplets[:, [0, 2, 1]].tolist()

    def build_vocab(self, train_triples, val_triples, test_triples):
        # datasets are already shipped with contiguous node IDs from 0 to N, so the total num ents is N+1
        all_triples = np.array(train_triples+val_triples+test_triples)
        train_nodes = np.unique(np.array(train_triples)[:, [0, 1]])
        val_nodes = np.unique(np.array(train_triples + val_triples)[:, [0, 1]])
        test_nodes = np.unique(np.array(train_triples + test_triples)[:, [0, 1]])
        num_entities = np.max(all_triples[:, [0, 1]]) + 1
        num_relations = np.max(all_triples[:, 2]) + 1

        ent_vocab = {i: i for i in range(num_entities)}
        rel_vocab = {i: i for i in range(num_relations)}

        return ent_vocab, rel_vocab, train_nodes, val_nodes, test_nodes

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        # num_entity in the inductive setup is different for different splits, take it from the relevant graph
        num_entity = self.num_entity_for_sample[index]
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": functional.as_mask(easy_answer, num_entity),
            "hard_answer": functional.as_mask(hard_answer, num_entity),
        }


@R.register("dataset.InductiveFB15k237CompExtendedEval")
class InductiveFB15k237CompExtendedEval(InductiveFB15k237Comp):

    """
    This dataset is almost equivalent to the original InductiveComp except that
    validation and test sets are training queries with a new (possibly larger) answer set
    being executed on a bigger validation or test graph

    We will load only the train_queries file and 3 different answer sets:
    1. train_queries_hard - original answers
    2. train_queries_val - answers to train queries over the validation graph (train + new val nodes and edges)
    3. train_queries_test - answers to train queries over the test graph (train + new test nodes and edges)

    The dataset is supposed to be used for evaluation/inference only,
    so make sure num_epochs is set to 0 in the config yaml file
    """

    def __init__(self, path, ratio, inverse_projection=False, query_types=None, union_type="DNF", verbose=1,
                 train_patterns=('1p', '2p', '3p', '2i', '3i', '2in', '3in', 'inp', 'pni', 'pin')):

        # Data Loader for inductive splits, ratio specifies the entity ratio of the training graph 0.1 - 0.9
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.inverse_projection = inverse_projection
        self.verbose = verbose
        query_types = query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, union_type)
                elif query_type[query_type.find("-") + 1:] != union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}
        self.union_type = union_type

        path = os.path.join(path, str(ratio))
        if not os.path.exists(path):
            raise FileNotFoundError(f"No dataset available at {path}")
            # utils.extract(zip_file)

        # Space of entities 0 ... N is split into 3 sets
        # Train node IDs: 0 ... K
        # Val inference ids: K ... K+M
        # Test inference ids: K+M .... N
        train_triplets = self.load_file(os.path.join(path, "train_graph.txt"))
        val_inference = self.load_file(os.path.join(path, "val_inference.txt"))
        test_inference = self.load_file(os.path.join(path, "test_inference.txt"))

        entity_vocab, relation_vocab, tr_nodes, vl_nodes, ts_nodes = self.build_vocab(train_triplets, val_inference,
                                                                                      test_inference)
        entity_vocab, inv_entity_vocab = self._standarize_vocab(entity_vocab, None)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, None)

        num_node = len(entity_vocab) if entity_vocab else None
        num_relation = len(relation_vocab) if relation_vocab else None

        # Training graph: only training triples
        self.train_graph = data.Graph(train_triplets, num_node=len(tr_nodes), num_relation=num_relation)

        # Validation graph: training triples (0..K) + new validation inference triples (K+1...K+M)
        self.valid_graph = data.Graph(train_triplets + val_inference, num_node=num_node, num_relation=num_relation)

        # Test graph: training triples (0..K) + new test inference triples (K+M+1... N)
        self.test_graph = data.Graph(train_triplets + test_inference, num_node=num_node, num_relation=num_relation)

        # Full graph (aux purposes)
        self.graph = data.Graph(train_triplets + val_inference + test_inference,
                                num_node=num_node, num_relation=num_relation)
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_entity_vocab = inv_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

        # Need those for evaluation
        self.valid_nodes = torch.tensor(vl_nodes, dtype=torch.long)
        self.test_nodes = torch.tensor(ts_nodes, dtype=torch.long)

        path = os.path.join(self.path, str(ratio))
        easy_answers = []
        hard_answers = []
        queries = []
        type_ids = []
        num_samples = []
        num_entity_for_sample = []
        max_query_length = 0

        # in this setup, we evaluate train queries on extended validation/test graphs
        # in extended graphs, training queries now have more answers
        # conceptually, all answers are "easy", but for eval purposes we load them as hard
        with open(os.path.join(path, "train_queries.pkl"), "rb") as fin:
            struct2queries = pickle.load(fin)

        #type2struct = {v: k for k, v in self.struct2type.items()}
        #filtered_training_structs = tuple([type2struct[x] for x in train_patterns])
        for split in ["train", "valid", "test"]:
            if split == "train":
                with open(os.path.join(path, "train_answers_hard.pkl"), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
            else:
                # load new answers
                with open(os.path.join(path, "train_answers_%s.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)

            query2easy_answers = defaultdict(lambda: defaultdict(set))

            num_sample = 0
            structs = sorted(struct2queries.keys(), key=lambda s: self.struct2type[s])
            if verbose:
                structs = tqdm(structs, "Loading %s queries" % split)
            for struct in structs:
                query_type = self.struct2type[struct]
                if query_type not in self.type2id:
                    continue

                struct_queries = struct2queries[struct]
                for i, query in enumerate(struct_queries):
                    # The dataset format is slightly different from BetaE's
                    #easy_answers.append(query2easy_answers[struct][i])
                    q_index = i if split != "train" else query
                    hard_answers.append(query2hard_answers[struct][q_index])
                    query = Query.from_nested(query)
                    max_query_length = max(max_query_length, len(query))
                    queries.append(query)
                    type_ids.append(self.type2id[query_type])
                num_sample += len(struct_queries)

            num_entity_for_sample += [getattr(self, "%s_graph" % split).num_node.item()] * num_sample
            num_samples.append(num_sample)

        self.queries = queries
        self.types = type_ids

        self.hard_answers = hard_answers
        self.easy_answers = [[] for _ in range(len(hard_answers))]
        self.num_samples = num_samples
        self.num_entity_for_sample = num_entity_for_sample
        self.max_query_length = max_query_length


class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    """Wrapper for inductive link prediction datasets where validation / test inference graphs extend the training.
    Can be used for pre-training inductive models on simple 1p link prediction objective.
    """

    def load_inductive_tsvs(self, path, verbose=0):

        train_triplets = self.load_file(os.path.join(path, "train_graph.txt"))
        val_inference = self.load_file(os.path.join(path, "val_inference.txt"))
        test_inference = self.load_file(os.path.join(path, "test_inference.txt"))

        entity_vocab, relation_vocab, tr_nodes, vl_nodes, ts_nodes = self.build_vocab(train_triplets, val_inference,
                                                                                      test_inference)
        inv_entity_vocab = {v: k for k, v in entity_vocab.items()}
        inv_relation_vocab = {v: k for k, v in relation_vocab.items()}

        val_predict = self.load_file(os.path.join(path, "val_predict.txt"))
        test_predict = self.load_file(os.path.join(path, "test_predict.txt"))

        num_samples = [len(x) for x in [train_triplets, val_predict, test_predict]]

        # train_entity_vocab, inv_train_entity_vocab = self._standarize_vocab(None, inv_train_entity_vocab)
        # test_entity_vocab, inv_test_entity_vocab = self._standarize_vocab(None, inv_test_entity_vocab)
        # relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.train_graph = data.Graph(train_triplets, num_node=len(entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = data.Graph(train_triplets + val_inference,
                                      num_node=len(entity_vocab), num_relation=len(relation_vocab))
        self.test_graph = data.Graph(train_triplets + test_inference,
                                     num_node=len(entity_vocab), num_relation=len(relation_vocab))
        self.graph = data.Graph(train_triplets + val_inference + test_inference,
                                     num_node=len(entity_vocab), num_relation=len(relation_vocab))

        # Need those for evaluation
        self.train_nodes = torch.tensor(tr_nodes, dtype=torch.long)
        self.valid_nodes = torch.tensor(vl_nodes, dtype=torch.long)
        self.test_nodes = torch.tensor(ts_nodes, dtype=torch.long)

        self.triplets = torch.tensor(train_triplets + val_predict + test_predict)
        self.num_samples = num_samples
        self.train_entity_vocab = self.val_entity_vocab = self.test_entity_vocab = entity_vocab

        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = self.inv_val_entity_vocab = self.inv_test_entity_vocab = inv_entity_vocab

        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def load_file(self, path):
        triplets = []
        with open(path) as fin:
            for line in fin:
                h, r, t = [int(x) for x in line.split()]
                triplets.append((h, t, r))

        return triplets

    def build_vocab(self, train_triples, val_triples, test_triples):
        # datasets are already shipped with contiguous node IDs from 0 to N, so the total num ents is N+1
        all_triples = np.array(train_triples+val_triples+test_triples)
        train_nodes = np.unique(np.array(train_triples)[:, [0, 1]])
        val_nodes = np.unique(np.array(train_triples + val_triples)[:, [0, 1]])
        test_nodes = np.unique(np.array(train_triples + test_triples)[:, [0, 1]])
        num_entities = np.max(all_triples[:, [0, 1]]) + 1
        num_relations = np.max(all_triples[:, 2]) + 1

        ent_vocab = {i: i for i in range(num_entities)}
        rel_vocab = {i: i for i in range(num_relations)}

        return ent_vocab, rel_vocab, train_nodes, val_nodes, test_nodes

@R.register("dataset.InductiveFB15k237DatasetLP")
class InductiveFB15k237DatasetLP(InductiveKnowledgeGraphDataset):
    url = "https://zenodo.org/record/7306046/files/%s.zip"

    md5 = {
        550: "e78bb9a7de9bd55813bb17f57941303c",
        300: "4db5c172acf83f676c9cf6589e033d7e",
        217: "9fde4563c619dc4d2b81af200cf7bc6b",
        175: "29ee1dbed7662740a2f001a0c6df8911",
        150: "61b545de8e5cdb04832f27842d8c0175",
        134: "cd8028c9674dc81f38cd17b03af43fe1",
        122: "272d2cc1e3f98f76d02daaf066f9d653",
        113: "e4ea60448e918c62779cfa757a096aa9",
        106: "6f9a1dcf22108074fb94a05b8377a173",
    }

    def __init__(self, path, ratio, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        # Download data if it's not there
        zip_file = utils.download(self.url % str(ratio), path, md5=self.md5[ratio])
        path = os.path.join(path, str(ratio))
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_inductive_tsvs(path, verbose=verbose)


@R.register("dataset.InductiveWikiKGDatasetLP")
class InductiveWikiKGDatasetLP(InductiveKnowledgeGraphDataset):
    url = "https://zenodo.org/record/7306046/files/wikikg.zip"

    md5 = "fa30b189436ab46a2ff083dd6a5e6e0b"

    def __init__(self, path, ratio=0.6, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        # Download data if it's not there
        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "wikikg")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pt(path, verbose=verbose)

    def load_pt(self, path, verbose=0):

        train_triplets = self.load_file(os.path.join(path, "train_graph.pt"))
        val_inference = self.load_file(os.path.join(path, "val_inference.pt"))
        test_inference = self.load_file(os.path.join(path, "test_inference.pt"))

        entity_vocab, relation_vocab, tr_nodes, vl_nodes, ts_nodes = self.build_vocab(train_triplets, val_inference,
                                                                                      test_inference)

        inv_entity_vocab = {v: k for k, v in entity_vocab.items()}
        inv_relation_vocab = {v: k for k, v in relation_vocab.items()}

        val_predict = self.load_file(os.path.join(path, "val_predict.pt"))
        test_predict = self.load_file(os.path.join(path, "test_predict.pt"))

        num_samples = [len(x) for x in [train_triplets, val_predict, test_predict]]

        self.train_graph = data.Graph(train_triplets, num_node=len(entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = data.Graph(torch.cat([train_triplets, val_inference], dim=0),
                                      num_node=len(entity_vocab), num_relation=len(relation_vocab))
        self.test_graph = data.Graph(torch.cat([train_triplets, test_inference], dim=0),
                                     num_node=len(entity_vocab), num_relation=len(relation_vocab))
        self.graph = data.Graph(torch.cat([train_triplets, val_inference, test_inference], dim=0),
                                num_node=len(entity_vocab), num_relation=len(relation_vocab))

        # Need those for evaluation
        self.train_nodes = torch.tensor(tr_nodes, dtype=torch.long)
        self.valid_nodes = torch.tensor(vl_nodes, dtype=torch.long)
        self.test_nodes = torch.tensor(ts_nodes, dtype=torch.long)

        self.triplets = torch.cat([train_triplets, val_predict, test_predict], dim=0)
        self.num_samples = num_samples
        self.train_entity_vocab = self.val_entity_vocab = self.test_entity_vocab = entity_vocab

        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = self.inv_val_entity_vocab = self.inv_test_entity_vocab = inv_entity_vocab

        self.inv_relation_vocab = inv_relation_vocab



    def load_file(self, path):
        triplets = torch.load(path, map_location="cpu")
        return torch.as_tensor(triplets[:, [0, 2, 1]])

    def build_vocab(self, train_triples, val_triples, test_triples):
        # datasets are already shipped with contiguous node IDs from 0 to N, so the total num ents is N+1

        train_nodes = train_triples[:, [0, 1]].unique()
        val_nodes = torch.cat([train_triples, val_triples], dim=0)[:, [0, 1]].unique()
        test_nodes = torch.cat([train_triples, test_triples], dim=0)[:, [0, 1]].unique()
        num_entities = torch.cat([train_triples, val_triples, test_triples], dim=0)[:, [0, 1]].max() + 1
        num_relations = torch.cat([train_triples, val_triples, test_triples], dim=0)[:, 2].max() + 1

        ent_vocab = {i: i for i in range(num_entities)}
        rel_vocab = {i: i for i in range(num_relations)}

        return ent_vocab, rel_vocab, train_nodes, val_nodes, test_nodes

