"""Lossless replay with a smaller host offload ceiling for this 32GiB host.

The original v7 runtime stays unchanged. Reducing host offload only leaves more
saved tensors on the GPU; dtype, strides, values, graph and 6GiB host reserve
are preserved. The worker verifies exact C0 logits AND raw gradients.
"""
from contextlib import contextmanager
from unittest.mock import patch
from vg_tta import desta3d_v2_output_anchor_layout_v5 as layout
from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
from vg_tta.desta3d_v3_actuation_full_vocab import full_coordinates

HOST_OFFLOAD_BYTES = 14 * 2**30
HOST_RESERVE_BYTES = 6 * 2**30


def replay_branch(*args, **kwargs):
    cleanup = release_free_host_arenas()
    assert cleanup['available_after_cleanup'] >= HOST_OFFLOAD_BYTES + HOST_RESERVE_BYTES, ('host reserve', cleanup)
    original = layout.activation_offload
    @contextmanager
    def offload(model):
        with original(model, limit_bytes=HOST_OFFLOAD_BYTES) as info:
            info.update(cleanup)
            info['host_reserve_bytes'] = HOST_RESERVE_BYTES
            yield info
    with full_coordinates(), patch.object(layout, 'activation_offload', offload):
        return layout.replay_branch(*args, **kwargs)
