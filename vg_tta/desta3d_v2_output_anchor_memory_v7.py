"""Return free libc arenas before measuring the next bounded offload reserve."""
from contextlib import contextmanager
from unittest.mock import patch
import ctypes
import gc
from vg_tta import desta3d_v2_output_anchor_layout_v5 as layout
from vg_tta.desta3d_v2_output_anchor_memory_v6 import host_available


def release_free_host_arenas():
    before=host_available()
    collected=gc.collect()
    libc=ctypes.CDLL(None)
    trim=libc.malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
    result=int(trim(0))
    return {'available_before_cleanup':before,'available_after_cleanup':host_available(),
            'gc_collected':collected,'malloc_trim_result':result}


def replay_branch(*args,**kwargs):
    original=layout.activation_offload
    cleanup=release_free_host_arenas();available=cleanup['available_after_cleanup'];limit=16*2**30
    assert available>=limit+6*2**30,('host reserve after release',cleanup)
    @contextmanager
    def offload(model):
        with original(model,limit_bytes=limit) as info:
            info.update(cleanup)
            info['host_reserve_bytes']=6*2**30
            yield info
    with patch.object(layout,'activation_offload',offload):
        return layout.replay_branch(*args,**kwargs)
