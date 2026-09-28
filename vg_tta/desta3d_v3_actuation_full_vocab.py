"""Only widen coordinate objective normalization to the true native vocabulary.

Native sampling and grammar are not patched. Event classes remain the original
registered endpoint classes. No extra structure loss, precision or prefix edit.
"""
from contextlib import contextmanager
from unittest.mock import patch
from vg_tta import desta3d_v2_output_anchor as base
from vg_tta.desta3d_v2_output_anchor_memory_v7 import replay_branch as memory_replay

def selected_full(logits,query,token_ids,coordinate_ids,original):
    if query and all(q in token_ids['ordered_time_tokens'] for q in query):
        if logits.shape[:2]!=(1,len(query)*6):raise ValueError('Native parallel block shape mismatch')
        return 'coordinate',logits.reshape(len(query),6,-1)[:,1:5].float()
    return original(logits,query,token_ids,coordinate_ids)

@contextmanager
def full_coordinates():
    original=base._selected_logits
    with patch.object(base,'_selected_logits',lambda *args:selected_full(*args,original)):
        yield

def capture_teacher(*args,**kwargs):
    with full_coordinates():return base.capture_teacher(*args,**kwargs)

def replay_branch(*args,**kwargs):
    with full_coordinates():return memory_replay(*args,**kwargs)
