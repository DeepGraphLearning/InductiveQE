import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm


logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def build_solver(cfg, dataset):

    # setting some model params depending on the task (pretraining / complex query)
    if "KnowledgeGraphCompletion" in cfg.task["class"]:
        cfg.task.model.num_relation = dataset.num_relation.item()
    if "LogicalQuery" in cfg.task["class"] and cfg.task.model['class'] != "HeuristicBaseline":
        cfg.task.model.model.num_relation = dataset.num_relation.item()
        # cfg.task.model.model.num_entity = dataset.num_entity.item()
    if cfg.task.model["class"] == "NodePiece":
        cfg.task.model.train_graph = dataset.train_graph
        cfg.task.model.valid_graph = dataset.valid_graph
        cfg.task.model.test_graph = dataset.test_graph

    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if "fast_test" in cfg:
        if comm.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid" % cfg.fast_test)
        g = torch.Generator()
        g.manual_seed(1024)
        valid_set = torch_data.random_split(valid_set, [cfg.fast_test, len(valid_set) - cfg.fast_test], generator=g)[0]
        #test_set = torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]
        if "wikikg" in cfg.task.model:
            # WikiKG valid/test splits have 600k triples, use the full set sizes only when you have A LOT of compute
            logger.warning("Limiting the number of test triples on WikiKG to %s " % cfg.fast_test)
            test_set = torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]

    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    # log more config args for wandb
    if "logger" in cfg.engine:
        if cfg.engine["logger"] == "wandb":
            solver.meter.logger.run.config.update(flatten_dict(cfg))

    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint)

    return solver


def save_embs(solver, mode, checkpoint):
    import pickle
    # materialize end save entity embeddings, relation embeddings, and the mapping dict for the CQD inference
    if comm.get_rank() == 0:
        logger.warning("Save embeddings to %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)
    graph = getattr(solver.model, "%s_graph" % mode)
    # we read nodes and relation types as is, so we can create the dicts right away
    e2id = {k: k for k in range(solver.model.num_entity)}
    r2id = {k: k for k in range(solver.model.num_relation)}
    # note that for relations we need to take the first half of the tensor since 2nd half is a duplicate, last = pad
    if solver.rank == 0:
        entities, rels = solver.model.model.encoder.materialize_embeddings(graph, mode)
        rels = rels[:solver.model.num_relation, :].detach().cpu()
        torch.save(entities.detach().cpu(), f"{checkpoint}_ents.pt")
        torch.save(rels, f"{checkpoint}_rels.pt")
        logger.warning("Saving e2id")
        with open(f"e2id.pkl", 'wb') as fout:
            pickle.dump(e2id, fout)
        logger.warning("Saving r2id")
        with open(f"r2id.pkl", 'wb') as fout:
            pickle.dump(r2id, fout)
        logger.warning("Saving Done")

    comm.synchronize()


def flatten_dict(d, parent_key: str = '', sep: str ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, int) or isinstance(v, bool) or isinstance(v, str) or isinstance(v, list) or isinstance(v, float):
            items.append((new_key, v))
    return dict(items)


class DebugHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if comm.get_rank() > 0:
            while True:
                pass

        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode="Plain", color_scheme="Linux", call_pdb=1)
        return self.instance(*args, **kwargs)


def setup_debug_hook():
    import sys
    sys.excepthook = DebugHook()