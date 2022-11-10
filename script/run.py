import os
import sys
import math
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from gnnqe import dataset, gnn, model, task, util, nodepiece, heuristic_baseline
from gnnqe import dataset, gnn, model, task, util, heuristic_baseline


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return None, None

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver, best_epoch


def test(cfg, solver):
    if "InductiveKnowledgeGraphCompletion" in cfg.task["class"]:
        solver.model.metric = ("mrr", "hits@1", "hits@3", "hits@10")
    else:
        solver.model.metric = ("mrr", "hits@1", "hits@3", "hits@10", "auroc")
    if not cfg.skip_eval_on_train:
        solver.model.split = "train"
        solver.evaluate("train")
    solver.model.split = "valid"
    solver.evaluate("valid")
    solver.model.split = "test"
    solver.evaluate("test")


if __name__ == "__main__":
    util.setup_debug_hook()
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    _, epoch = train_and_validate(cfg, solver)
    test(cfg, solver)

    if cfg.save_embs:
        solver.model.split = "test"
        if "wikikg" in cfg.task.model:
            solver.model.cpu()
        util.save_embs(solver, "test", f"epoch_{epoch}")
