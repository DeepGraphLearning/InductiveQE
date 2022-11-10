import torch
from torch import nn
from collections import defaultdict
from tqdm import tqdm

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import Stack


@R.register("model.HeuristicBaseline")
class HeuristicBaseline(nn.Module, core.Configurable):

    stack_size = 2

    def __init__(self):
        super(HeuristicBaseline, self).__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def execute(self, graph, query, all_loss=None, metric=None):
        """Execute queries on the graph."""
        # we use stacks to execute postfix notations
        # check out this tutorial if you are not familiar with the algorithm
        # https://www.andrew.cmu.edu/course/15-121/lectures/Stacks%20and%20Queues/Stacks%20and%20Queues.html
        batch_size = len(query)
        # we execute a neural model and a symbolic model at the same time
        # the symbolic model is used for traversal dropout at training time
        # the elements in a stack are fuzzy sets
        self.stack = Stack(batch_size, self.stack_size, graph.num_node, device=self.device)
        # instruction pointer
        self.IP = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        all_sample = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        op = query[all_sample, self.IP]

        while not op.is_stop().all():
            is_operand = op.is_operand()
            is_intersection = op.is_intersection()
            is_union = op.is_union()
            is_negation = op.is_negation()
            is_projection = op.is_projection()
            if is_operand.any():
                h_index = op[is_operand].get_operand()
                self.apply_operand(is_operand, h_index, graph.num_node)
            if is_intersection.any():
                self.apply_intersection(is_intersection)
            if is_union.any():
                self.apply_union(is_union)
            if is_negation.any():
                self.apply_negation(is_negation)
            # only execute projection when there are no other operations
            # since projection is the most expensive and we want to maximize the parallelism
            if not (is_operand | is_negation | is_intersection | is_union).any() and is_projection.any():
                r_index = op[is_projection].get_operand()
                self.apply_projection(is_projection, graph, r_index, all_loss=all_loss, metric=metric)
            op = query[all_sample, self.IP]

        if (self.stack.SP > 1).any():
            raise ValueError("More operands than expected")

    def forward(self, graph, query, all_loss=None, metric=None):
        self.execute(graph, query, all_loss=all_loss, metric=metric)

        # prob is either 0 or 1
        # modify prob with some random noise
        t_prob = self.stack.pop()
        rand = torch.rand_like(t_prob)
        t_prob = (t_prob * 2 + rand) / 3
        # convert probability to logit for compatibility reasons
        t_logit = ((t_prob + 1e-10) / (1 - t_prob + 1e-10)).log()
        return t_logit

    def apply_operand(self, mask, h_index, num_node):
        h_prob = functional.one_hot(h_index, num_node)
        self.stack.push(mask, h_prob)
        self.IP[mask] += 1

    def apply_negation(self, mask):
        x = self.stack.pop(mask)
        self.stack.push(mask, 1 - x)
        self.IP[mask] += 1

    def apply_intersection(self, mask):
        y = self.stack.pop(mask)
        x = self.stack.pop(mask)
        self.stack.push(mask, torch.min(x, y))
        self.IP[mask] += 1

    def apply_union(self, mask):
        y = self.stack.pop(mask)
        x = self.stack.pop(mask)
        self.stack.push(mask, torch.max(x, y))
        self.IP[mask] += 1

    def apply_projection(self, mask, graph, r_index, all_loss=None, metric=None):
        any = -torch.ones_like(r_index)
        pattern = torch.stack([any, any, r_index], dim=-1)
        edge_index, num_match = graph.match(pattern)
        t_index = graph.edge_list[edge_index, 1]
        sample_index = functional._size_to_index(num_match)

        x = torch.zeros(mask.sum(), graph.num_node, device=self.device)
        x[sample_index, t_index] = 1
        self.stack.pop(mask)
        self.stack.push(mask, x)
        self.IP[mask] += 1
