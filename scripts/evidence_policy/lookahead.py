"""Exact finite-horizon reference consequences on an already frozen bank.

This is a tensor-only CPU calculation, not a model, bank builder or optimizer.
The first action is forced; the next H-1 discrete actions follow the fixed
instantaneous reference distribution. All valid continuations are integrated,
including the derivative of their probabilities and their memory states.
Only a loop over depth is used. Nodes and branches at each depth are batched.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _get(bank, name, fallback=None):
    if isinstance(bank, Mapping):
        return bank.get(name, fallback)
    return getattr(bank, name, fallback)


def _tensor(value, name, dtype):
    _require(torch.is_tensor(value), name+' must already be a tensor')
    _require(value.device.type == 'cpu' and value.dtype == dtype,
             name+' has an incompatible CPU dtype')
    if value.is_floating_point():
        _require(bool(torch.isfinite(value).all()), name+' must be finite')
    return value


def _prepare(bank, nodes, memory, previous_probs, prev_features):
    features = _tensor(_get(bank, 'features'), 'features', torch.float64)
    _require(features.ndim == 3 and features.shape[1] == 3,
             'Features must have shape [T,3,D]')
    T, K, D = features.shape
    _require(T > 0 and D > 0, 'A nonempty frozen bank is required')
    unary = _get(bank, 'u', _get(bank, 'projection'))
    b = dict(features=features, u=_tensor(unary, 'u', torch.float64),
             guard=_tensor(_get(bank, 'guard'), 'guard', torch.float64),
             link=_tensor(_get(bank, 'link'), 'link', torch.float64),
             confidence=_tensor(_get(bank, 'confidence'), 'confidence', torch.float64),
             active=_tensor(_get(bank, 'active'), 'active', torch.bool),
             valid=_tensor(_get(bank, 'valid'), 'valid', torch.bool))
    expected = dict(u=(T,K), guard=(T,K), link=(T-1,K,K),
                    confidence=(T,), active=(T,), valid=(T,K))
    for name, shape in expected.items():
        _require(b[name].shape == shape, name+' has an incompatible shape')
    _require(bool(b['valid'][:,0].all()), 'The E action must always be valid')
    _require(bool(((b['confidence'] >= 0) & (b['confidence'] <= 1)).all()),
             'Confidence must lie in [0,1]')
    _require(bool(((b['u'] >= 0) & (b['u'] <= 1)).all()), 'u must lie in [0,1]')
    # Inputs are the policy's prepared bank, whose feature vectors have already
    # been normalized. Re-normalizing here would change H=1 rounding.
    b = {name: value.detach() for name, value in b.items()}
    nodes = _tensor(nodes, 'nodes', torch.long)
    memory = _tensor(memory, 'memory', torch.float64)
    previous_probs = _tensor(previous_probs, 'previous_probs', torch.float64)
    _require(nodes.ndim == 1, 'nodes must have shape [N]')
    N = len(nodes)
    _require(memory.shape == (N,D) and previous_probs.shape == (N,K),
             'Visited memory/previous probabilities have incompatible shapes')
    _require(bool(((nodes >= 0) & (nodes < T)).all()), 'A visited node is out of range')
    _require(bool((previous_probs >= 0).all()), 'Previous probabilities must be nonnegative')
    _require(torch.allclose(previous_probs.sum(-1), torch.ones(N,dtype=memory.dtype),
                            atol=1e-12, rtol=0),
             'Previous probabilities must sum to one')
    if prev_features is not None:
        prev_features = _tensor(prev_features, 'prev_features', torch.float64)
        _require(prev_features.shape == (N,D), 'prev_features has an incompatible shape')
    return b, nodes, memory, previous_probs


def effective_valid(bank, nodes):
    """The fixed event gate restricts an inactive node to action E."""
    valid = _get(bank, 'valid')[nodes]
    active = _get(bank, 'active')[nodes]
    E_only = torch.zeros_like(valid)
    E_only[:,0] = True
    return torch.where(active[:,None], valid, E_only)


def instantaneous_scores(bank, nodes, memory, previous_probs):
    """Original reference r, with no prior or confidence multiplier.

    ``memory`` and bank features use the original prepared normalization.
    The previous selected feature has no independent term in reference r.
    """
    features = _get(bank, 'features')[nodes]
    memory_cos = (memory[:,None,:]*features).sum(-1)
    # A leading zero edge makes node zero explicit without an extra state loop.
    links = _get(bank, 'link')
    zero = links.new_zeros((1,3,3))
    incoming = torch.cat((zero,links),dim=0)[nodes]
    previous_link = torch.bmm(previous_probs[:,None,:],incoming).squeeze(1)
    return (_get(bank,'u',_get(bank,'projection'))[nodes]
            + .25*previous_link + .1*memory_cos - .1*_get(bank,'guard')[nodes])


def _advance(bank, nodes, memory, actions):
    selected_feature = bank['features'][nodes,actions]
    selected_u = bank['u'][nodes,actions]
    omega = .2*bank['confidence'][nodes]*selected_u
    omega = torch.where(bank['active'][nodes],omega,torch.zeros_like(omega))
    proposed = F.normalize((1-omega[:,None])*memory + omega[:,None]*selected_feature,
                           dim=-1,eps=1e-12)
    # The original update performs no normalization/write when omega == zero.
    memory_after = torch.where((omega != 0)[:,None],proposed,memory)
    previous_after = F.one_hot(actions,num_classes=3).to(torch.float64)
    return memory_after,previous_after


def exact_lookahead(bank, nodes, memory, previous_probs, *, horizon=4,
                    prev_features=None):
    """Return exact eligible Q_H and tensor operation counts.

    Args:
        bank: Prepared FP64 CPU bank (mapping or attribute container). Its
            features must have the same normalization as the calling policy.
        nodes: Physical indices ``[N]`` for any batch of visited states.
        memory: Differentiable current states ``[N,D]``.
        previous_probs: Differentiable previous action probabilities ``[N,3]``.
        horizon: H in [1,4], including the forced current action. At the video
            endpoint the remaining sum is truncated, never divided by H.
        prev_features: Optional shape-checked API companion. The original
            reference does not use this student-only feature separately.

    Inactive starting rows have Q=0 and only E support; their adaptation loss
    weight is zero. For active starts, inactive nodes inside the future window
    contribute rho=0, advance the previous action to E, and do not write memory.
    Invalid starting actions have Q=0 and must be excluded with effective_valid.

    Reference probabilities are softmax(r/.2) over valid actions, without pi0.
    Future rho=active*r has no extra confidence factor. Confidence enters the
    existing memory write and the caller's outer loss. No state/probability is
    detached: Direct can differentiate the complete finite discrete sum.

    Costs count actual evaluated branch states and valid discrete branches.
    ``continuation_paths[N,3]`` counts the enumerated leaf continuations for
    each forced action (at most 27 at H=4); no stochastic approximation is used.
    """
    _require(isinstance(horizon,int) and not isinstance(horizon,bool) and 1 <= horizon <= 4,
             'The exact reference horizon must be an integer in [1,4]')
    b,nodes,memory,previous_probs = _prepare(bank,nodes,memory,previous_probs,prev_features)
    N,K = len(nodes),3
    active = b['active'][nodes]
    valid = effective_valid(b,nodes)
    r = instantaneous_scores(b,nodes,memory,previous_probs)
    q = torch.where(active[:,None] & valid,r,torch.zeros_like(r))
    q_flat = q.reshape(-1)
    leaves = torch.zeros((N*K,),dtype=torch.long)
    evaluated = torch.zeros(horizon,dtype=torch.long)
    action_branches = torch.zeros(horizon,dtype=torch.long)
    transitions = torch.zeros(horizon,dtype=torch.long)
    evaluated[0] = N
    root_valid = valid & active[:,None]
    root_row,root_action = torch.nonzero(root_valid,as_tuple=True)
    root_index = root_row*K+root_action
    action_branches[0] = len(root_row)
    terminal = (nodes[root_row]+1 >= b['features'].shape[0]) | (horizon == 1)
    leaves.index_add_(0,root_index[terminal],torch.ones_like(root_index[terminal]))
    continuing = ~terminal
    # Only enumerate eligible active roots. Inactive starting rows are excluded
    # by definition; they cannot contribute to the confidence-weighted loss.
    parents = root_row[continuing]
    actions = root_action[continuing]
    origin = root_index[continuing]
    branch_nodes = nodes[parents]
    branch_memory,branch_previous = _advance(b,branch_nodes,memory[parents],actions)
    transitions[0] = len(parents)
    branch_nodes = branch_nodes+1
    mass = memory.new_ones(len(parents))
    for depth in range(1,horizon):
        if len(branch_nodes) == 0:
            break
        evaluated[depth] = len(branch_nodes)
        local_valid = effective_valid(b,branch_nodes)
        local_r = instantaneous_scores(b,branch_nodes,branch_memory,branch_previous)
        probs = F.softmax((local_r/.2).masked_fill(~local_valid,-torch.inf),dim=-1)
        rho = torch.where(b['active'][branch_nodes,None],local_r,torch.zeros_like(local_r))
        expected_rho = (probs*rho).sum(-1)
        q_flat = q_flat.index_add(0,origin,mass*expected_rho)
        parent_index,action = torch.nonzero(local_valid,as_tuple=True)
        action_branches[depth] = len(parent_index)
        branch_origin = origin[parent_index]
        next_nodes = branch_nodes[parent_index]
        terminal = (next_nodes+1 >= b['features'].shape[0]) | (depth == horizon-1)
        leaves.index_add_(0,branch_origin[terminal],torch.ones_like(branch_origin[terminal]))
        continuing = ~terminal
        p, a = parent_index[continuing],action[continuing]
        # Expand only physically valid continuations; probabilities (including
        # their gradients) remain attached to each complete branch.
        mass = mass[p]*probs[p,a]
        branch_memory,branch_previous = _advance(b,branch_nodes[p],branch_memory[p],a)
        branch_nodes = branch_nodes[p]+1
        origin = branch_origin[continuing]
        transitions[depth] = len(p)
    costs = dict(input_states=torch.tensor(N,dtype=torch.long),
        eligible_start_states=active.sum(),
        excluded_inactive_start_states=(~active).sum(),
        branch_states_by_depth=evaluated,
        valid_action_branches_by_depth=action_branches,
        state_transitions_by_depth=transitions,
        branch_state_evaluations=evaluated.sum(),
        candidate_score_evaluations=K*evaluated.sum(),
        valid_action_branches=action_branches.sum(),
        state_transitions=transitions.sum(),
        continuation_paths=leaves.reshape(N,K),
        enumerated_continuation_paths=leaves.sum())
    return dict(q=q_flat.reshape(N,K),effective_valid=valid,costs=costs,
                horizon=horizon,inactive_start_constant_omitted=True,
                exact_discrete_expectation=True,reference_tau=.2)


# Descriptive alias for callers that prefer the feedback name.
lookahead_feedback = exact_lookahead
