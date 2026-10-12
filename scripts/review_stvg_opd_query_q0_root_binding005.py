"""Keep the completed CPU receipt distinct from the paired-contrast label.

This is a postseal root-audit binding repair.  The pinned Q0 scientific runner,
scorer, data, readouts and all original statistical formulae remain unchanged.
"""
import inspect
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import context


def repaired_run():
    context()
    from scripts import review_stvg_opd_query_q0_v1 as original
    source = inspect.getsource(original.run)
    old = "verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');c=read(BASE/'CPU_COMPLETION.json')"
    new = "verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');cpu_completion=read(BASE/'CPU_COMPLETION.json')"
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "and c['cells']==64 and c['GT_read']"
    assert source.count(old) == 1
    source = source.replace(old, "and cpu_completion['cells']==64 and cpu_completion['GT_read']")
    old = "for p,h in c['files'].items():"
    assert source.count(old) == 2
    source = source.replace(old, "for p,h in cpu_completion['files'].items():")
    namespace = dict(original.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['run']


def run():
    repaired_run()()


if __name__ == '__main__':
    run()
