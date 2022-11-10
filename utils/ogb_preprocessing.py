import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import click

from typing import List, Union
from collections import defaultdict, Counter
from oos_splitting import DatasetPreprocess
from create_queries import generate_queries
import random
from config_ratios import CONFIG_RATIOS

np.random.seed(42)
random.seed(42)

DATA_PATH= "fb15k237"
VERSION="betae"

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

UNION = -1
NEGATION = -2

any = 1 << 25
projection = 1 << 26
intersection = 1 << 27
union = 1 << 28
negation = 1 << 29
stop = 1 << 30

config=CONFIG_RATIOS

def preprocess_oos(ds_name: str, fpath: str, ratio: List[float]=[0.4, 0.3, 0.3], limit_query_types: str =None, union_type: str = "DNF"):

    """

    The function splits the original transductive query datasets into the out-of-sample mode
    """

    ds_path = Path(fpath)

    print(f"Ratios: {ratio}")
    query_types = struct2type.values()
    new_query_types = []
    for query_type in query_types:
        if "u" in query_type:
            if "-" not in query_type:
                query_type = "%s-%s" % (query_type, union_type)
            elif query_type[query_type.find("-") + 1:] != union_type:
                continue
        new_query_types.append(query_type)
    id2type = sorted(new_query_types)
    type2id = {t: i for i, t in enumerate(id2type)}

    # Load train / val / test triples
    tr_ratio, vl_ratio, ts_ratio = ratio
    heads, rels, tails = [], [], []
    for split in ["train.pt", "valid_triples.pt", "test_triples.pt"]:
        tr = torch.load(ds_path / split)
        heads.append(tr['head'])
        rels.append(tr['relation'])
        tails.append(tr['tail'])

    triples = np.vstack([
        np.concatenate(heads),
        np.concatenate(rels),
        np.concatenate(tails)
    ]).T

    num_ents = np.unique(triples[:, [0, 2]]).shape[0]
    num_rels = np.unique(triples[:, 1]).shape[0]
    print(f"Dataset: {ds_name}, num ents: {num_ents}, rels: {num_rels}")

    splitter = DatasetPreprocess(triples, num_ents, num_rels, vl_ratio+ts_ratio, 0.5)
    splitter.make_dataset()

    print(f"Training graph: {splitter.ent_splits[0]} nodes, {len(splitter.old_triples)} triples")
    print(f"Val graph: {splitter.ent_splits[1]} nodes, {len(splitter.val_inference)} triples, {len(splitter.val_predict)} missing")
    print(f"Test graph: {splitter.ent_splits[2]} nodes, {len(splitter.test_inference)} triples, {len(splitter.test_predict)} missing")

    # save(Path("../data"), tr_ratio, splitter,
    #      None,
    #      None,
    #      None
    # )
    # return

    queries, answers = sample_queries(
        splitter.old_triples,
        splitter.new_val_triples,
        splitter.new_test_triples,
        config['wikikg2'],
        target_query_types=limit_query_types,
    )

    # save
    save(Path("../projects/ogb/dataset/ogbl_wikikg2/split/qa"), tr_ratio, splitter,
         (queries[0], None, answers[0]),
         (queries[1], answers[1], answers[2]),
         (queries[2], answers[3], answers[4]),
         target_query_types=limit_query_types,
    )


    print("Processing complete")
    return 0


def sample_queries(train_triples, val_triples, test_triples, params, target_query_types=None):
    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    query_structures = [
        [e, [r]],
        [e, [r, r]],
        [e, [r, r, r]],
        [[e, [r]], [e, [r]]],
        [[e, [r]], [e, [r]], [e, [r]]],
        [[e, [r, r]], [e, [r]]],
        [[[e, [r]], [e, [r]]], [r]],
        # negation
        [[e, [r]], [e, [r, n]]],
        [[e, [r]], [e, [r]], [e, [r, n]]],
        [[e, [r, r]], [e, [r, n]]],
        [[e, [r, r, n]], [e, [r]]],
        [[[e, [r]], [e, [r, n]]], [r]],
        # union
        [[e, [r]], [e, [r]], [u]],
        [[[e, [r]], [e, [r]], [u]], [r]]
    ]
    query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
    if target_query_types is not None:
        print(f"Mining only {target_query_types} queries")
        target_query_types = target_query_types.split(",")
        query_filtered = [q for q in query_names if q in target_query_types]
        query_structures = [query_structures[i] for i, q in enumerate(query_names) if q in target_query_types]
        query_names = query_filtered
        print("Filtered query structures to mine: ", query_structures)

    max_ans_num = 1e3

    train_queries, val_queries, test_queries = defaultdict(set), defaultdict(set), defaultdict(set)
    train_hard_answers, val_easy_answers, val_hard_answers, test_easy_answers, test_hard_answers = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)

    for i in range(len(query_names)):

        tr, vl, ts = generate_queries(
            [train_triples, val_triples, test_triples],
            [query_structures[i]],
            params[query_names[i]],
            max_ans_num, True, True, True, query_names[i], None
        )

        # train_queries[str(query_structures[i])] = tr[0]
        # val_queries[str(query_structures[i])] = vl[0]
        # test_queries[str(query_structures[i])] = ts[0]
        train_queries.update(tr[0])
        train_hard_answers.update(tr[2])
        val_queries.update(vl[0])
        val_easy_answers.update(vl[1])
        val_hard_answers.update(vl[2])
        test_queries.update(ts[0])
        test_easy_answers.update(ts[1])
        test_hard_answers.update(ts[2])

    return (train_queries, val_queries, test_queries), (train_hard_answers, val_easy_answers, val_hard_answers, test_easy_answers, test_hard_answers)


def filter_answers(nodes_to_keep: Union[list, tuple], answers: List[set]) -> List[set]:
    result = []
    if type(nodes_to_keep) == tuple:
        to_keep = set(nodes_to_keep[0]).union(set(nodes_to_keep[1]))
    else:
        to_keep = set(nodes_to_keep)

    # keep only entities-answers from the input list, others do not exist in the given graph
    for ans in answers:
        filtered = set(ans).intersection(to_keep) if len(ans) != 0 else []
        result.append(filtered)

    return result



def remove_queries_nodes(nodes_to_remove: list, queries):
    # all entities are encoded as IDs >=0
    result = []
    to_remove = set(nodes_to_remove)
    for query in queries:
        q_entities = set([e for e in query if e >= 0])
        if len(q_entities.intersection(to_remove)) > 0:
            continue
        result.append(query)

    return result


def remove_queries_rels(rels_to_remove: list, queries):
    to_remove = set(rels_to_remove)
    result = []

    for query in queries:
        rels = [decode_rel(e) for e in query if e < 0]
        rels = [r for r in rels if r is not None]
        if len(set(rels).intersection(to_remove)) > 0:
            continue
        result.append(query)

    return result

def decode_rel(token):
    is_stop = token == stop
    is_operand = ~is_stop & (token >= 0)
    is_negation = ~is_operand & ~is_stop & (-token & negation > 0)
    is_intersection = ~is_operand & ~is_stop & (-token & intersection > 0)
    is_union = ~is_operand & ~is_stop & (-token & union > 0)
    is_projection = ~is_operand & ~is_stop & (-token & projection > 0)

    if not (bool(is_operand) | bool(is_negation) | bool(is_intersection) | bool(is_union) | is_stop) and is_projection:
        r_index = -token & ~projection
        return r_index
    else:
        return None

def filter_queries(entities, queries):
    result = []

    if type(entities) == tuple:
        train_e, test_e = entities
        total = set(train_e).union(set(test_e))
        train_e, test_e = set(train_e), set(test_e)
    else:
        train_e, test_e = entities, None
        total = set(train_e)
        train_e = set(train_e)

    for query in queries:
        q_entities = set([e for e in query if e >= 0])
        if test_e is not None:
            # if we found a query with val/test nodes, make sure that all other nodes are only in the total set, st
            # there will be no test entities in val queries and vice versa
            if len(q_entities.intersection(test_e)) > 0:
                if q_entities.issubset(total):
                    result.append(query)
        else:
            # we are only filtering the training queries
            if q_entities.issubset(train_e):
                result.append(query)

    return result


def to_postfix_notation(query):

    notation = []

    is_projection = True
    for op in query[1]:
        if isinstance(op, tuple):
            is_projection = False
            break

    if is_projection: # chain of projection / negation
        entity, unary_ops = query
        if isinstance(entity, int):
            notation.append(entity)
            is_variable = False
        else:
            notation = to_postfix_notation(entity)
            is_variable = True
        for op in unary_ops:
            if op == NEGATION: # negation
                notation.append(-negation)
            else: # projection
                if is_variable:
                    notation.append(any)
                    notation.append(-(projection | (op ^ 1)))
                    notation.append(-intersection)
                notation.append(-(projection | op))
                is_variable = True
    else:
        if query[-1] == (UNION,): # union
            op = union
            query = query[:-1]
        else: # intersection
            op = intersection
        for i, sub_query in enumerate(query):
            sub_notation = to_postfix_notation(sub_query)
            notation += sub_notation
            if i > 0:
                notation.append(-op)

    return notation

def save(path: Path,
         tr_ratio: float,
         splitter: DatasetPreprocess,
         train_data: tuple,
         val_data: tuple,
         test_data: tuple,
         target_query_types: str = None):
    p = path / f"{tr_ratio}"
    p.mkdir(exist_ok=True, parents=True)

    # save the graph
    if not (p / "train_graph.pt").exists():
        write_triples(p / "train_graph.pt", splitter.old_triples)
        write_triples(p / "val_inference.pt", splitter.new_val_triples[0])
        write_triples(p / "val_predict.pt", splitter.new_val_triples[1])
        write_triples(p / "test_inference.pt", splitter.new_test_triples[0])
        write_triples(p / "test_predict.pt", splitter.new_test_triples[1])
        #return
        write_stats(p / "stats.txt", splitter.ent_splits, splitter.global_r2id)
        write_og(p / "og_mappings.pkl", splitter.global_e2id, splitter.global_r2id)

    # save queries
    write_queries(p, "train", target_query_types, *train_data)
    write_queries(p, "valid", target_query_types, *val_data)
    write_queries(p, "test", target_query_types, *test_data)

def write_triples(fname, triples):
    torch.save(np.array(triples), fname)
    print(f"Writing {fname} done")

def write_og(fname, e2id, r2id):
    output = {"e2id": e2id, "r2id": r2id}
    pickle.dump(output, open(fname, "wb"))

def write_queries(fname, dtype, target_query_types, train_q, tr_easy, tr_hard):
    #pickle.dump(train_q, open(fname / f"{dtype}_queries_tdrug.pkl", "wb"))

    qname = f"{dtype}_queries"
    easyname = f"{dtype}_answers_easy"
    hardname = f"{dtype}_answers_hard"
    if target_query_types is not None:
        qname += f"_{target_query_types}"
        easyname += f"_{target_query_types}"
        hardname += f"_{target_query_types}"
    pickle.dump(train_q, open(fname / f"{qname}.pkl", "wb"))
    if tr_easy is not None:
        pickle.dump(tr_easy, open(fname / f"{easyname}.pkl", "wb"))
    pickle.dump(tr_hard, open(fname / f"{hardname}.pkl", "wb"))

    print("Done writing queries")

def write_stats(fname, num_ents, r2id):
    with open(fname, "w") as fout:
        fout.write(f"numentity: {sum(num_ents)}\n")
        fout.write(f"numrelations: {len(r2id)}\n")
        fout.write(f"entities in train: {num_ents[0]}\n")
        fout.write(f"entities in validation: {num_ents[1]}\n")
        fout.write(f"entities in test: {num_ents[2]}\n")


@click.command()
@click.option('--start', type=int, default=60)
@click.option('--end', type=int, default=70)
@click.option('--dataset', type=str, default="wikikg2")
@click.option('--path', type=str, default="../projects/ogb/dataset/ogbl_wikikg2/split/time")
@click.option('--queries', type=str, default=None)  # or "1p,2p,3p,2i, so on"
def main(
        start: int,
        end: int,
        dataset: str,
        path: str,
        queries: str,
):
    limit = 100 if end ==0 else end
    for i in range(start, limit, 10):
        ratio = float(i / 100)
        preprocess_oos(dataset, path, [ratio, (1-ratio) / 2, (1-ratio)/2], limit_query_types=queries)


if __name__ == "__main__":
    main()

