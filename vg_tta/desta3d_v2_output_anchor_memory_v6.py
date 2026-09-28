"""Same differentiable replay; 16GiB host offload for the larger target grid."""
from contextlib import contextmanager
from unittest.mock import patch
from vg_tta import desta3d_v2_output_anchor_layout_v5 as layout


def host_available():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('host available memory not measurable')


def replay_branch(*args,**kwargs):
    original=layout.activation_offload
    available=host_available();limit=16*2**30
    assert available>=limit+6*2**30,'host reserve before larger offload'
    @contextmanager
    def offload(model):
        with original(model,limit_bytes=limit) as info:
            info['host_available_before']=available
            info['host_reserve_bytes']=6*2**30
            yield info
    with patch.object(layout,'activation_offload',offload):
        return layout.replay_branch(*args,**kwargs)
