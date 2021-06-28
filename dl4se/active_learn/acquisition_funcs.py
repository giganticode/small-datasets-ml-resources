import torch
import torch.nn.functional as F

import math

def least_conf(probs, targets=None):
    most_conf, _ = torch.max(probs, dim=1)
    num_labels = probs.shape[1]
    z = num_labels / (num_labels - 1)
    return z * (1.0 - most_conf)


def margin_conf(probs, targets=None):
    topk, _ = torch.topk(probs, 2, dim=1, sorted=True)
    return 1.0 - (topk[:, 0] - topk[:, 1])


def ratio_conf(probs, targets=None, negate=True):
    topk, _ = torch.topk(probs, 2, dim=1, sorted=True)
    return topk[:, 1] / topk[:, 0]


def random_conf(probs, targets=None):
    return torch.rand((probs.shape[0]))


def entropy(probs, targets=None):
    products = -torch.log2(probs) * probs
    num_labels = probs.shape[1]
    return torch.sum(products, dim=1) / math.log2(num_labels)


def cross_entropy(logits, targets):
    num_labels = logits.shape[1]
    return F.cross_entropy(logits, targets, reduction='none') / math.log2(num_labels)