"""Subject marginals and point-set overlap on a laminar forest (children precede parents)."""
from collections import deque

import numpy as np


def overlap_sum(parent, size, mass):
    """Apply the point-set Jaccard kernel in two passes, including each node once."""
    below = size * mass
    above = np.zeros(len(parent))
    for node, p in enumerate(parent):
        if p >= 0:
            below[p] += below[node]
    for node in reversed(range(len(parent))):
        p = parent[node]
        if p >= 0:
            above[node] = above[p] + mass[p] / size[p]
    return below / size + size * above


def disjoint_pairs(parent):
    """The ordered-witness kernel needs a mask for pairs, including the same node."""
    allowed = np.ones((len(parent), len(parent)), dtype=bool)
    for node in range(len(parent)):
        ancestor = node
        while ancestor >= 0:
            allowed[node, ancestor] = allowed[ancestor, node] = False
            ancestor = parent[ancestor]
    return allowed


class SideEvidence:
    """Sum evidence from disjoint sibling subtrees past a horizontal interval boundary."""

    def __init__(self, parent, stop, view):
        n = len(parent)
        self.visible = np.flatnonzero(np.isfinite(view[:, 0]))
        self.view = view
        reach = np.ones(n)
        for node in reversed(range(n)):
            p = parent[node]
            if p >= 0:
                reach[node] = reach[p] * (1. - stop[p])
        self.occupancy = reach * stop
        self.stop = stop

        # Group siblings into binary sums. This lets us read "all other children"
        # using only positive additions, even when one sibling dominates the sum.
        groups = [deque() for _ in range(n + 1)]
        for node, p in enumerate(parent):
            groups[p if p >= 0 else n].append(node)
        self.up = [-1] * (n + 1)
        self.sibling = [-1] * (n + 1)
        self.split = list(1. - stop) + [1.]
        for p, children in enumerate(groups):
            while len(children) > 1:
                a, b = children.popleft(), children.popleft()
                group = len(self.up)
                self.up[a] = self.up[b] = group
                self.sibling[a], self.sibling[b] = b, a
                self.up.append(-1)
                self.sibling.append(-1)
                self.split.append(1.)
                children.append(group)
            if children:
                self.up[children[0]] = p

    def marginal(self, subject, reference, side):
        # Reflect right-of queries so both directions use the same strict test.
        lo, hi = self.view[:, 0], self.view[:, 2]
        left, right = (lo, hi) if side < 0 else (-hi, -lo)
        visible = self.visible
        refs = visible[np.argsort(-left[visible], kind='stable')]
        subjects = visible[np.argsort(-right[visible], kind='stable')]
        subtree = np.zeros(len(self.up))
        out = np.zeros(len(subject))
        cursor = 0
        for i in subjects:
            while cursor < len(refs) and left[refs[cursor]] > right[i]:
                node = refs[cursor]
                cursor += 1
                value = self.stop[node] * reference[node]
                while node >= 0 and value > 0:
                    subtree[node] += value
                    node = self.up[node]
                    if node >= 0:
                        value *= self.split[node]
            total = 0.
            node = i
            while node >= 0:
                other = self.sibling[node]
                if other >= 0:
                    total += subtree[other]
                node = self.up[node]
            out[i] = self.occupancy[i] * subject[i] * total
        return out
