"""Native last-layer prior on the original physical grid; no layer aggregation."""
import numpy as np
from .temporal import probabilities, mass_table

def native_prior(records, logits, frame_ids):
    ids, q = mass_table(probabilities(records, logits))
    if ids != list(map(int, frame_ids)):
        raise ValueError('Native physical time grid differs from the request')
    return q

def from_capture(cap):
    # Old sealed caches carry every layer. Read only their exact final-layer
    # logits; this compatibility path does not evaluate a six-layer head.
    logits = cap.get('native_logits')
    if logits is None:
        logits = [z[-1] for z in cap['logits']]
    q = native_prior(cap['records'], logits, cap['frame_ids'])
    if 'native_prior' in cap and not np.array_equal(q, cap['native_prior']):
        raise ValueError('Cached native prior changed')
    return q
