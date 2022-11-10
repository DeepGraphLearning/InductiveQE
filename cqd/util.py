import numpy as np
import random
import torch
import time

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def load_file(path):
    triplets = []
    with open(path) as fin:
        for line in fin:
            h, r, t = [int(x) for x in line.split()]
            triplets.append((h, t, r))

    return triplets


def get_unique_nodes(train_triples, val_inf, test_inf):
    val_nodes = np.unique(np.array(train_triples + val_inf)[:, [0, 1]])
    test_nodes = np.unique(np.array(train_triples + test_inf)[:, [0, 1]])

    return torch.tensor(val_nodes, dtype=torch.long), torch.tensor(test_nodes, dtype=torch.long)

def get_unique_nodes_wikikg(train_triples, val_inf, test_inf):
    train_nodes = np.unique(train_triples[:, [0, 2]])
    val_nodes = np.unique(np.concatenate([train_triples, val_inf])[:, [0, 2]])
    test_nodes = np.unique(np.concatenate([train_triples, test_inf])[:, [0, 2]])

    return torch.tensor(train_nodes, dtype=torch.long), torch.tensor(val_nodes, dtype=torch.long), torch.tensor(test_nodes, dtype=torch.long)