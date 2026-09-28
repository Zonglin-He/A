"""One-factor ablation: keep the registered time anchor, remove coordinate KL."""
from vg_tta.desta3d_v2_output_anchor_tta import adapt_output_anchor, teacher_support
from vg_tta.desta3d_v2_output_anchor_memory_v7 import replay_branch


def temporal_teacher_support(native, trace):
    return [b for b in teacher_support(native, trace) if b == 'event']


def adapt_temporal_anchor(*args, **kwargs):
    # Underlying runner and parameter updates remain identical. The teacher
    # capture still includes spatial output for matching and diagnostics.
    enabled = args[6] if len(args) > 6 else kwargs['enabled']
    if any(branch != 'event' for branch in enabled):
        raise ValueError('temporal-only ablation cannot optimize spatial output KL')
    kwargs.setdefault('replay', replay_branch)
    report = adapt_output_anchor(*args, **kwargs)
    report['output_factor'] = 'time-only; coordinate coefficient exactly zero; no coefficient renormalization'
    return report
