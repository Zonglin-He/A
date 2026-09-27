"""Explicit arrival-state semantics, separate from episodic production code."""
from methods.decota_final_simplified_v1.tensors import detached

ARMS = ('E', 'O-all', 'O-split')
QUERY = 'spatial.query_residual'


def arrival(source, previous, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == 'E' or previous is None:
        return detached(source)
    if set(previous) != set(source):
        raise ValueError('State schema changed')
    result = {k: v.detach().clone().to(source[k]) for k, v in previous.items()}
    if arm == 'O-split':
        result[QUERY] = source[QUERY].detach().clone()
    return result
