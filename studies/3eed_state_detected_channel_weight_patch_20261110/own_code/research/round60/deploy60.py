"""Deployment-facing detect -> patch -> replay -> restore, no experiment inputs.

This wrapper takes the existing frozen 3EED head, original ABC last-hidden
activations/tokens/masks, and original geometry/UNKNOWN transport. It does not
load development IDs, aggregate fixed-patch statistics, task labels or teachers.
R60 does not adopt either tested detector; this is the reproducible interface.
"""
import numpy as np
from mechanism60 import detect,replay

def detect_patch_replay(head,original_view_states,original_association,mechanism='noise'):
    if mechanism not in ('noise','common'):
        raise ValueError('A registered detector must be named explicitly')
    W=head[4].weight.detach().cpu().numpy().astype(float)
    b=head[4].bias.detach().cpu().numpy().astype(float)
    gates,diagnostic=detect(original_view_states,original_association,W,b)
    gate=gates[mechanism]
    scores,status=replay(head,original_view_states,original_association,gate)
    return {'scores':scores,'selected_original_candidate':int(np.argmax(scores)),
            'column_attenuation':gate,'status':status,
            'cross_sampling_support':diagnostic['support_mass']}
