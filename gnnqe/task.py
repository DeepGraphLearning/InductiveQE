import math
import torch

from torch.nn import functional as F
from torch.utils import data as torch_data

from torch_scatter import scatter_add, scatter_mean

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("task.LogicalQuery")
class LogicalQuery(tasks.Task, core.Configurable):
    """
    Logical query task.

    Parameters:
        model (nn.Module): logical query model
        criterion (str, list or dict, optional): training criterion(s). Only ``bce`` is available for now.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mrr``, ``hits@K``, ``mape`` and ``spearmanr``.
        adversarial_temperature (float, optional): temperature for self-adversarial negative sampling.
            Set ``0`` to disable self-adversarial negative sampling.
        sample_weight (bool, optional): whether to weight each query by its number of answers
    """

    _option_members = ["criterion", "metric", "query_type_weight"]

    def __init__(self, model, criterion="bce", metric=("mrr",), adversarial_temperature=0, sample_weight=False):
        super(LogicalQuery, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.adversarial_temperature = adversarial_temperature
        self.sample_weight = sample_weight

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.id2type = dataset.id2type
        self.type2id = dataset.type2id

        self.register_buffer("fact_graph", dataset.fact_graph)
        self.register_buffer("graph", dataset.graph)

        return train_set, valid_set, test_set

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                is_positive = target > 0.5
                is_negative = target <= 0.5
                num_positive = is_positive.sum(dim=-1)
                num_negative = is_negative.sum(dim=-1)
                neg_weight = torch.zeros_like(pred)
                neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        logit = pred[is_negative] / self.adversarial_temperature
                        neg_weight[is_negative] = functional.variadic_softmax(logit, num_negative)
                else:
                    neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            if self.sample_weight:
                sample_weight = target.sum(dim=-1).float()
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            else:
                loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict_and_target(self, batch, all_loss=None, metric=None):
        query = batch["query"]
        type = batch["type"]
        easy_answer = batch["easy_answer"]
        hard_answer = batch["hard_answer"]

        pred = self.model(self.fact_graph, query, all_loss, metric)
        if all_loss is None:
            target = (type, easy_answer, hard_answer)
            ranking = self.batch_evaluate(pred, target)
            # answer set cardinality prediction
            prob = F.sigmoid(pred)
            num_pred = (prob * (prob > 0.5)).sum(dim=-1)
            num_easy = easy_answer.sum(dim=-1)
            num_hard = hard_answer.sum(dim=-1)
            return (ranking, num_pred), (type, num_easy, num_hard)
        else:
            target = easy_answer.float()

        return pred, target

    def batch_evaluate(self, pred, target):
        type, easy_answer, hard_answer = target

        num_easy = easy_answer.sum(dim=-1)
        num_hard = hard_answer.sum(dim=-1)
        num_answer = num_easy + num_hard
        answer2query = functional._size_to_index(num_answer)
        order = pred.argsort(dim=-1, descending=True)
        range = torch.arange(self.num_entity, device=self.device)
        ranking = scatter_add(range.expand_as(order), order, dim=-1)
        easy_ranking = ranking[easy_answer]
        hard_ranking = ranking[hard_answer]
        # unfiltered rankings of all answers
        answer_ranking = functional._extend(easy_ranking, num_easy, hard_ranking, num_hard)[0]
        order_among_answer = functional.variadic_sort(answer_ranking, num_answer)[1]
        order_among_answer = order_among_answer + (num_answer.cumsum(0) - num_answer)[answer2query]
        ranking_among_answer = scatter_add(functional.variadic_arange(num_answer), order_among_answer)

        # filtered rankings of all answers
        ranking = answer_ranking - ranking_among_answer + 1
        ends = num_answer.cumsum(0)
        starts = ends - num_hard
        hard_mask = functional.multi_slice_mask(starts, ends, ends[-1])
        # filtered rankings of hard answers
        ranking = ranking[hard_mask]

        return ranking

    def evaluate(self, pred, target):
        ranking, num_pred = pred
        type, num_easy, num_hard = target

        metric = {}
        for _metric in self.metric:
            if _metric == "mrr":
                answer_score = 1 / ranking.float()
                query_score = functional.variadic_mean(answer_score, num_hard)
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                answer_score = (ranking <= threshold).float()
                query_score = functional.variadic_mean(answer_score, num_hard)
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric == "mape":
                query_score = (num_pred - num_easy - num_hard).abs() / (num_easy + num_hard).float()
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric == "spearmanr":
                type_score = []
                for i in range(len(self.id2type)):
                    mask = type == i
                    score = metrics.spearmanr(num_pred[mask], num_easy[mask] + num_hard[mask])
                    type_score.append(score)
                type_score = torch.stack(type_score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            score = type_score.mean()
            is_neg = torch.tensor(["n" in t for t in self.id2type], device=self.device)
            is_epfo = ~is_neg
            name = tasks._get_metric_name(_metric)
            for i, query_type in enumerate(self.id2type):
                metric["[%s] %s" % (query_type, name)] = type_score[i]
            if is_epfo.any():
                epfo_score = functional.masked_mean(type_score, is_epfo)
                metric["[EPFO] %s" % name] = epfo_score
            if is_neg.any():
                neg_score = functional.masked_mean(type_score, is_neg)
                metric["[negation] %s" % name] = neg_score
            metric[name] = score

        return metric

    def visualize(self, batch):
        query = batch["query"]
        return self.model.visualize(self.fact_graph, self.graph, query)



@R.register("task.InductiveLogicalQuery")
class InductiveLogicalQuery(LogicalQuery):

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.id2type = dataset.id2type
        self.type2id = dataset.type2id
        self.register_buffer("train_graph", dataset.train_graph)
        self.register_buffer("valid_graph", dataset.valid_graph)
        self.register_buffer("test_graph", dataset.test_graph)

        # add explicit val/test entities for constrained evaluation
        self.register_buffer("valid_nodes", dataset.valid_nodes)
        self.register_buffer("test_nodes", dataset.test_nodes)

        return train_set, valid_set, test_set

    def predict_and_target(self, batch, all_loss=None, metric=None):
        query = batch["query"]
        type = batch["type"]
        easy_answer = batch["easy_answer"]
        hard_answer = batch["hard_answer"]

        graph = getattr(self, "%s_graph" % self.split)
        pred = self.model(graph, query, all_loss, metric)
        if all_loss is None:
            # test, in case of GPU OOM
            restrict_nodes = getattr(self, "%s_nodes" % self.split).cpu() if self.split == "valid" or self.split == "test" else None
            pred = pred.cpu()
            type = type.cpu()
            easy_answer = easy_answer.cpu()
            hard_answer = hard_answer.cpu()
            target = (type, easy_answer, hard_answer)
            ranking, answer_ranking = self.batch_evaluate(pred, target, limit_nodes=restrict_nodes)
            num_easy = easy_answer.sum(dim=-1)
            num_hard = hard_answer.sum(dim=-1)
            return ranking, (type, answer_ranking, num_easy, num_hard)
        else:
            target = easy_answer.float()

        return pred, target

    def batch_evaluate(self, pred, target, limit_nodes=None):

        """
        :param limit_nodes: if not None, use ranks for only those specified nodes, mask others to -inf
        """
        type, easy_answer, hard_answer = target

        num_easy = easy_answer.sum(dim=-1)
        num_hard = hard_answer.sum(dim=-1)
        num_answer = num_easy + num_hard
        answer2query = functional._size_to_index(num_answer)
        num_entity = pred.shape[-1]

        if limit_nodes is not None:
            # print(f"Keeping only {len(limit_nodes)} nodes out of {num_entity}")
            keep_mask = functional.as_mask(limit_nodes, num_entity)
            pred[:, ~keep_mask] = float('-inf')

        order = pred.argsort(dim=-1, descending=True)
        ranking = scatter_add(torch.arange(num_entity).expand_as(order), order, dim=-1)

        easy_ranking = ranking[easy_answer]
        hard_ranking = ranking[hard_answer]
        answer_ranking = functional._extend(easy_ranking, num_easy, hard_ranking, num_hard)[0]
        order_among_answer = functional.variadic_sort(answer_ranking, num_answer)[1]
        order_among_answer = order_among_answer + (num_answer.cumsum(0) - num_answer)[answer2query]
        ranking_among_answer = scatter_add(functional.variadic_arange(num_answer), order_among_answer)

        # filtered ranking
        ranking = answer_ranking - ranking_among_answer + 1
        ends = num_answer.cumsum(0)
        starts = ends - num_hard
        hard_mask = functional.multi_slice_mask(starts, ends, ends[-1])
        ranking = ranking[hard_mask]

        return ranking, answer_ranking

    def evaluate(self, pred, target):
        if pred.dtype == torch.float:
            type, easy_answer, hard_answer = target

            num_easy = easy_answer.sum(dim=-1)
            num_hard = hard_answer.sum(dim=-1)
            num_answer = num_easy + num_hard
            answer2query = functional._size_to_index(num_answer)
            order = pred.argsort(dim=-1, descending=True)
            num_entity = pred.shape[-1]
            ranking = scatter_add(torch.arange(num_entity).expand_as(order), order, dim=-1)
            easy_ranking = ranking[easy_answer]
            hard_ranking = ranking[hard_answer]
            answer_ranking = functional._extend(easy_ranking, num_easy, hard_ranking, num_hard)[0]
            order_among_answer = functional.variadic_sort(answer_ranking, num_answer)[1]
            order_among_answer = order_among_answer + (num_answer.cumsum(0) - num_answer)[answer2query]
            ranking_among_answer = scatter_add(functional.variadic_arange(num_answer), order_among_answer)

            # filtered ranking
            ranking = answer_ranking - ranking_among_answer + 1
            ends = num_answer.cumsum(0)
            starts = ends - num_hard
            hard_mask = functional.multi_slice_mask(starts, ends, ends[-1])
            ranking = ranking[hard_mask]
        else:
            ranking = pred
            type, answer_ranking, num_easy, num_hard = target

        metric = {}
        for _metric in self.metric:
            if _metric == "mrr":
                answer_score = 1 / ranking.float()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                answer_score = (ranking <= threshold).float()
            elif _metric == "auroc":
                ends = (num_easy + num_hard).cumsum(0)
                starts = ends - num_hard
                target = functional.multi_slice_mask(starts, ends, len(answer_ranking)).float()
                answer_score = variadic_area_under_roc(answer_ranking, target, num_easy + num_hard)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            if _metric == "auroc":
                mask = (num_easy > 0) & (num_hard > 0)
                query_score = answer_score[mask]
                type_score = scatter_mean(query_score, type[mask], dim_size=len(self.id2type))
            else:
                query_score = functional.variadic_mean(answer_score, num_hard)
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            score = type_score.mean()
            is_epfo = torch.tensor(["n" not in t for t in self.id2type])
            name = tasks._get_metric_name(_metric)
            for i, query_type in enumerate(self.id2type):
                metric["[%s] %s" % (query_type, name)] = type_score[i]
                # metric["[%s (main CC)] %s" % (query_type, name)] = main_type_score[i]
            if not is_epfo.all():
                epfo_score = functional.masked_mean(type_score, is_epfo)
                metric["[EPFO] %s" % name] = epfo_score
            metric[name] = score

        return metric

# Pre-training objective on inductive splits with disjoint validation and test entities sets
@R.register("tasks.InductiveKnowledgeGraphCompletion")
class InductiveKnowledgeGraphCompletion(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, sample_weight=True,
                 batched_eval=False):
        super(InductiveKnowledgeGraphCompletion, self).__init__(
            model, criterion, metric, num_negative, margin, adversarial_temperature, strict_negative,
            sample_weight=sample_weight)
        self.batched_eval = batched_eval

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity.item()
        self.num_relation = dataset.num_relation.item()

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        self.register_buffer("train_graph", dataset.train_graph)
        self.register_buffer("valid_graph", dataset.valid_graph)
        self.register_buffer("test_graph", dataset.test_graph)
        self.register_buffer("fact_graph", dataset.train_graph)

        # add explicit val/test entities for constrained evaluation
        self.register_buffer("train_nodes", dataset.train_nodes)
        self.register_buffer("valid_nodes", dataset.valid_nodes)
        self.register_buffer("test_nodes", dataset.test_nodes)

        return train_set, valid_set, test_set


    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)
        graph = getattr(self, "%s_graph" % self.split)

        if all_loss is None:
            # test
            all_index = torch.arange(graph.num_node, device=self.device)
            t_preds = []
            h_preds = []
            #print("Scoring all tails")
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, mode=self.split)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            #print("Scoring all heads")
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, mode=self.split)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            # in case of GPU OOM
            pred = torch.stack([t_pred.cpu(), h_pred.cpu()], dim=1)
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, mode=self.split)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        graph = getattr(self, "%s_graph" % self.split)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = functional._size_to_index(num_t_truth)
        t_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = functional._size_to_index(num_h_truth)
        h_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()

    def evaluate(self, pred, target):
        mask, target = target

        restrict_nodes = getattr(self, "%s_nodes" % self.split).cpu()
        num_entity = pred.shape[-1]

        if restrict_nodes is not None:
            keep_mask = functional.as_mask(restrict_nodes, num_entity)
            pred[:, :, ~keep_mask] = float('-inf')

        pos_pred = pred.gather(-1, target.unsqueeze(-1))

        if not self.batched_eval:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1)  # + 1
        else:
            ranking = torch.zeros((pos_pred.shape[0], pos_pred.shape[1]), dtype=torch.long)
            for idx in torch.arange(pred.shape[-1]).split(self.num_negative * 10):
                ranking += torch.sum((pos_pred <= pred[:, :, idx]) & mask[:, :, idx], dim=-1)

        # adjust for train
        if self.split == "train":
            ranking += 1

        print("Computing metrics")
        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                values = _metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (ranking - 1).float() / mask.sum(dim=-1)
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample negatives
                        num_comb = math.factorial(num_sample) / math.factorial(i) / math.factorial(num_sample - i)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i))
                    score = score.mean()
                else:
                    score = (ranking <= threshold).float().mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score
        print("Metrics computed")
        return metric


def variadic_area_under_roc(pred, target, size):
    """
    Area under receiver operating characteristic curve (ROC) for sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        pred (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(B,)`.
        size (Tensor): size of sets of shape :math:`(N,)`
    """
    index2graph = functional._size_to_index(size)
    _, order = functional.variadic_sort(pred, size, descending=True)
    cum_size = (size.cumsum(0) - size)[index2graph]
    target = target[order + cum_size]
    total_hit = functional.variadic_sum(target, size)
    total_hit = total_hit.cumsum(0) - total_hit
    hit = target.cumsum(0) - total_hit[index2graph]
    hit = torch.where(target == 0, hit, torch.zeros_like(hit))
    all = functional.variadic_sum((target == 0).float(), size) * \
            functional.variadic_sum((target == 1).float(), size)
    auroc = functional.variadic_sum(hit, size) / (all + 1e-10)
    return auroc

