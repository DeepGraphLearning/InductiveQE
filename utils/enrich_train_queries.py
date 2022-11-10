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
from create_queries import achieve_answer, tuple2list, list2tuple
from preprocessing import write_queries


np.random.seed(42)
random.seed(42)

DATA_PATH = "fb15k237"

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
        #((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
        #((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
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

def construct_graph_from_triples(triples):
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for triple in triples:
        s,r,o = triple
        ent_out[s][r].add(o)
        ent_in[o][r].add(s)

    return ent_in, ent_out

def load_file(fname):
    triples = []
    with open(fname, "r") as fin:
        for line in fin:
            h,r,t = [int(x) for x in line.split()]
            triples.append([h,r,t])

    return triples

def load_graph(path, ratio):

    src = Path(path)

    train = load_file(src / f"{str(ratio)}" / "train_graph.txt")
    val_inference = load_file(src / f"{str(ratio)}" / "val_inference.txt")
    test_inference = load_file(src / f"{str(ratio)}" / "test_inference.txt")

    val_in, val_out = construct_graph_from_triples(train+val_inference)
    test_in, test_out = construct_graph_from_triples(train+test_inference)

    return (val_in, val_out), (test_in, test_out)

def load_test_graphs(path, ratio):

    src = Path(path)

    train = load_file(src / f"{str(ratio)}" / "train_graph.txt")
    val_inference = load_file(src / f"{str(ratio)}" / "val_inference.txt")
    val_predict = load_file(src / f"{str(ratio)}" / "val_predict.txt")
    test_inference = load_file(src / f"{str(ratio)}" / "test_inference.txt")
    test_predict = load_file(src / f"{str(ratio)}" / "test_predict.txt")

    small_test_in, small_test_out = construct_graph_from_triples(train + test_inference)
    full_test_in, full_test_out = construct_graph_from_triples(train + test_inference + test_predict)

    return (train, val_inference, val_predict, test_inference, test_predict), (small_test_in, small_test_out), (full_test_in, full_test_out)

def load_queries(path, ratio):
    src = Path(path)
    queries = pickle.load(open(src / f"{str(ratio)}" / "train_queries.pkl", "rb"))
    answers = pickle.load(open(src / f"{str(ratio)}" / "train_answers_hard.pkl", "rb"))

    return queries, answers

def load_train_queries_over_test(path, ratio):
    src = Path(path)
    queries = pickle.load(open(src / f"{str(ratio)}" / "train_queries.pkl", "rb"))
    old_answers = pickle.load(open(src / f"{str(ratio)}" / "train_answers_test.pkl", "rb"))
    new_answers = pickle.load(open(src / f"{str(ratio)}" / "train_answers_test_qneg.pkl", "rb"))
    return queries, old_answers, new_answers

def load_test_queries(path, ratio):
    src = Path(path)
    queries = pickle.load(open(src / f"{str(ratio)}" / "test_queries.pkl", "rb"))
    answers_easy = pickle.load(open(src / f"{str(ratio)}" / "test_answers_easy.pkl", "rb"))
    answers_hard = pickle.load(open(src / f"{str(ratio)}" / "test_answers_hard.pkl", "rb"))

    return queries, answers_easy, answers_hard

def enrich_train_queries(path: str, ratio: float):
    """This function obtains answers on train queries evaluated on bigger graphs (train+val_inf)) and (train+test_inf)
    It is possible that certain queries would have more answers than in the pure training set
    We want to estimate the 'faithfulness' aka completeness / generalization of QE models onto a bigger graph

    """
    val_set, test_set = load_graph(path, ratio)

    queries, answers = load_queries(path, ratio)
    val_query_answers = {}
    test_query_answers = {}

    for qtype in struct2type:
        # if struct2type[qtype] != "2in":
        #     continue
        qs = queries[qtype]
        val_query_answers[qtype] = []
        test_query_answers[qtype] = []
        new_val, new_test, total = 0, 0, len(qs)
        for query in tqdm(qs):
            orig_answers = len(answers[qtype][query])
            val_answers = achieve_answer(tuple2list(query), val_set[0], val_set[1])
            test_answers = achieve_answer(tuple2list(query), test_set[0], test_set[1])

            val_query_answers[qtype].append(val_answers)
            test_query_answers[qtype].append(test_answers)

            if len(val_answers) != orig_answers:
                new_val += 1
            if len(test_answers) != orig_answers:
                new_test += 1

        print(f"{struct2type[qtype]}: More answers in validation graph: {new_val} / {total}")
        print(f"{struct2type[qtype]}: More answers in test graph: {new_test} / {total}")

    write_answers(val_query_answers, test_query_answers, path, ratio)

    print(f"Done for {ratio}")


def write_answers(val_answers, test_answers, path, ratio):
    p = Path(path)
    pickle.dump(val_answers, open(p / f"{str(ratio)}" / "train_answers_val.pkl", "wb"))
    pickle.dump(test_answers, open(p / f"{str(ratio)}" / "train_answers_test.pkl", "wb"))

def write_test_answers(test_answers, path, ratio):
    p = Path(path)
    pickle.dump(test_answers, open(p / f"{str(ratio)}" / "train_answers_test.pkl", "wb"))

def write_neg_answers(test_answers, path, ratio):
    p = Path(path)
    pickle.dump(test_answers, open(p / f"{str(ratio)}" / "train_answers_test_qneg.pkl", "wb"))


def enrich_train_negative_queries(path: str, ratio: float):
    """This function obtains answers on train queries evaluated on the test graphs (train+test_inf)
    Hotfix for the sampled datasets

    """
    val_set, test_set = load_graph(path, ratio)

    queries, answers = load_queries(path, ratio)
    val_query_answers = {}
    test_query_answers = {}

    for qtype in struct2type:
        if struct2type[qtype] not in ["2in", "3in", "inp", "pin", "pni"]:
            continue
        qs = queries[qtype]
        val_query_answers[qtype] = []
        test_query_answers[qtype] = []
        new_val, new_test, total = 0, 0, len(qs)
        for query in tqdm(qs):
            orig_answers = len(answers[qtype][query])
            test_answers = achieve_answer(tuple2list(query), test_set[0], test_set[1])

            test_query_answers[qtype].append(test_answers)
            if len(test_answers) != orig_answers:
                new_test += 1

        print(f"{struct2type[qtype]}: More answers in test graph: {new_test} / {total}")

    write_neg_answers(test_query_answers, path, ratio)

    print(f"Done for {ratio}")


def remine_test_negative_queries(path: str, ratio: float):
    """
    This function obtains new answers of test queries
    Hotfix for the sampled datasets

    """
    graphs, small_test, full_test = load_test_graphs(path, ratio)
    params = CONFIG_RATIOS

    queries, answers_easy, answers_hard = load_test_queries(path, ratio)

    test_queries = defaultdict(set)
    test_easy_answers, test_hard_answers = defaultdict(set), defaultdict(set)

    for qtype in struct2type:
        if struct2type[qtype] not in ["2in", "3in", "inp", "pin", "pni"]:
            continue

        tr, vl, ts = generate_queries(
            [graphs[0], (graphs[1], graphs[2]), (graphs[3], graphs[4])],
            [tuple2list(qtype)],
            params[ratio][struct2type[qtype]],
            1000, False, False, True, struct2type[qtype], None
        )

        test_queries.update(ts[0])
        test_easy_answers.update(ts[1])
        test_hard_answers.update(ts[2])

        # replace bad queries in the original dumps
        queries[qtype] = ts[0][qtype]
        answers_easy[qtype] = ts[1][qtype]
        answers_hard[qtype] = ts[2][qtype]

    write_queries(Path(path) / f"{ratio}", "new_neg_test", test_queries, test_easy_answers, test_hard_answers)
    write_queries(Path(path) / f"{ratio}", "new_test", queries, answers_easy, answers_hard)

    print(f"Done for {ratio}")


def replace_new_train_negs(path: str, ratio: float):
    """
    This function replaces incorrect answers for train negative queries over the test graph
    with newly mined answers saved in train_answers_test_qneg
    """
    print("Updating answers for train queries over the test graph")
    queries, old_ans, new_ans = load_train_queries_over_test(path, ratio)

    for qtype in struct2type:
        if struct2type[qtype] not in ["2in", "3in", "inp", "pin", "pni"]:
            continue

        # replace new answers
        old_ans[qtype] = new_ans[qtype]

    write_test_answers(old_ans, path, ratio)
    print(f"Done for {ratio}")

@click.group()
def main():
    pass


@main.command("mine_all")
@click.option('--start', type=int, default=10)
@click.option('--end', type=int, default=0)
@click.option('--path', type=str, default="../data")
def mine_all(
        start: int,
        end: int,
        path: str,
):
    limit = 100 if end == 0 else end
    for i in range(start, limit, 10):
        ratio = float(i / 100)
        enrich_train_queries(path, ratio)


@main.command("mine_neg")
@click.option('--start', type=int, default=10)
@click.option('--end', type=int, default=0)
@click.option('--path', type=str, default="../data")
def mine_neg(
        start: int,
        end: int,
        path: str,
):
    limit = 100 if end == 0 else end
    for i in range(start, limit, 10):
        ratio = float(i / 100)
        enrich_train_negative_queries(path, ratio)

@main.command("mine_test")
@click.option('--start', type=int, default=10)
@click.option('--end', type=int, default=0)
@click.option('--path', type=str, default="../data")
def mine_neg(
        start: int,
        end: int,
        path: str,
):
    limit = 100 if end == 0 else end
    for i in range(start, limit, 10):
        ratio = float(i / 100)
        remine_test_negative_queries(path, ratio)

@main.command("replace_train_queries")
@click.option('--start', type=int, default=10)
@click.option('--end', type=int, default=0)
@click.option('--path', type=str, default="../data")
def mine_neg(
        start: int,
        end: int,
        path: str,
):
    limit = 100 if end == 0 else end
    for i in range(start, limit, 10):
        ratio = float(i / 100)
        replace_new_train_negs(path, ratio)


if __name__ == "__main__":
    main()
