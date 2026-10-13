"""Finite real-output candidate selection; no Jacobian or gradient surrogate."""
import numpy as np
import torch
from vg_tta.c1_enabling_tricks_v1 import QUERY
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

ARMS = ['A', 'LN', 'Query', 'Half']
RHO = .9

def candidate_states(before, after):
    assert list(before) == list(after) and QUERY in before
    assert sum(v.numel() for v in before.values()) == 1792
    assert before[QUERY].numel() == 256
    assert all(v.dtype == torch.float32 for v in before.values())
    return {
        'A': {n: v.clone() for n, v in after.items()},
        'LN': {n: (before[n] if n == QUERY else v).clone() for n, v in after.items()},
        'Query': {n: (v if n == QUERY else before[n]).clone() for n, v in after.items()},
        'Half': {n: before[n] + .5 * (v - before[n]) for n, v in after.items()},
    }

def metrics(boxes, current, entering, positions, targets, unvisited):
    gain = float((aligned_iou(boxes[positions].double(), targets.double()) -
                  aligned_iou(current[positions].double(), targets.double())).mean()) if positions else None
    drift = float((boxes[unvisited].double() - entering[unvisited].double()).square().sum() / len(unvisited)) if unvisited else None
    return {'G': gain, 'D': drift}

def choose(rows, rho=RHO):
    a = rows['A']
    if a['G'] is None: return 'A', [], 'empty_observed'
    if a['G'] <= 0: return 'A', [], 'nonpositive_original_gain'
    if a['D'] is None: return 'A', [], 'empty_unobserved'
    feasible = [n for n in ARMS if rows[n]['G'] >= rho * a['G']]
    assert 'A' in feasible
    selected = min(feasible, key=lambda n: (rows[n]['D'], ARMS.index(n)))
    return selected, feasible, 'positive_original_gain'

def useful(rows, feasible):
    if not feasible: return []
    tolerance = 1e-10 + 64*np.finfo(np.float32).eps*rows['A']['D']
    return [n for n in feasible if n != 'A' and rows['A']['D']-rows[n]['D'] > tolerance]

def contracts():
    cases = [
      ({'A':{'G':1.,'D':4.},'LN':{'G':.91,'D':2.},'Query':{'G':.89,'D':0.},'Half':{'G':.9,'D':3.}},'LN'),
      ({'A':{'G':0.,'D':4.},'LN':{'G':1.,'D':0.},'Query':{'G':1.,'D':0.},'Half':{'G':1.,'D':0.}},'A'),
      ({n:{'G':None,'D':1.} for n in ARMS},'A'),
      ({n:{'G':1.,'D':None} for n in ARMS},'A'),
      ({n:{'G':1.,'D':1.} for n in ARMS},'A'),
    ]
    for rows, expected in cases: assert choose(rows)[0] == expected
    before={QUERY:torch.arange(256,dtype=torch.float32),**{f'LN{i}':torch.ones(256) for i in range(6)}}
    after={n:v+.125 for n,v in before.items()};cs=candidate_states(before,after)
    for n in before:
        assert torch.equal(cs['A'][n],after[n])
        assert torch.equal(cs['LN'][n],before[n] if n==QUERY else after[n])
        assert torch.equal(cs['Query'][n],after[n] if n==QUERY else before[n])
        assert torch.equal(cs['Half'][n],before[n]+.0625)
    assert useful(cases[0][0],choose(cases[0][0])[1]) == ['LN','Half']
    return dict(status='pass',scope='synthetic selection/unknown/nonpositive/tie/block-mask/half-step contracts; not GPU feasibility',count=6,CPU_only=True,GT_read=False)
