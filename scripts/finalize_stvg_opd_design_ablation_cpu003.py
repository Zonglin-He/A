"""Additive CPU projection which annotates each immutable result on first write.

The original scientific runtime and the preserved CPU002 source remain intact.
No fit, statistic, bootstrap, score, or table rounding rule changes here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import finalize_stvg_opd_design_ablation_v1 as original
from scripts import finalize_stvg_opd_design_ablation_cpu002 as legacy
from scripts.stvg_opd_design_ablation_common_v1 import BASE, PUB, read, write, sha, verify

REV = BASE / 'CPU_projection_revision003'
NOTE = ('Historical table readback corrected two display values: shuffled-feedback '
        'Cross VidSTG 9.517 → 9.516, and frozen-source Corrupt HC2 29.093 → 29.094. '
        'All original predictions, per-query rows and underlying scores remain unchanged.')


def prepare():
    verify()
    legacy.prepare()
    record = dict(status='additive_CPU_projection_revision_locked',
        original_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        historical_CPU002_runtime_sha256=sha(legacy.REV / 'REVISION_RUNTIME.json'),
        pins={rel: sha(ROOT / rel) for rel in [
            'scripts/finalize_stvg_opd_design_ablation_cpu003.py',
            'scripts/finalize_stvg_opd_design_ablation_cpu002.py',
            'scripts/finalize_stvg_opd_design_ablation_v1.py']},
        original_table_readback_sha256=sha(legacy.REV / 'ORIGINAL_TABLE_READBACK.json'),
        scope='CPU projection metadata added on immutable first write; original math, rows, statistics and GPU pins unchanged',
        CPU_only=True, new_model_calls=0, new_GT_read=False, time=time.time())
    rp = REV / 'REVISION_RUNTIME.json'
    if rp.exists():
        previous = read(rp)
        for key in record:
            if key != 'time': assert previous[key] == record[key], key
    else:
        write(rp, record)
    return read(rp)


def run():
    revision = prepare()
    assert not (BASE / 'CPU_COMPLETION.json').exists(), 'Preserve any partial CPU result before a separate recovery.'
    assert not (PUB / 'COMPONENT_TABLE.json').exists(), 'Immutable new projection outputs must be fresh.'
    original.existing_table = legacy.existing_table
    initial_write = original.write
    def annotated_first_write(path, value):
        path = Path(path)
        if path == PUB / 'COMPONENT_TABLE.json':
            value = dict(value, historical_display_corrections=read(legacy.REV / 'ORIGINAL_TABLE_READBACK.json')['corrections'],
                CPU_revision_runtime_sha256=sha(REV / 'REVISION_RUNTIME.json'))
        if path == BASE / 'CPU_COMPLETION.json':
            for name in ['REPORT.md', 'COMPONENT_TABLE.md']:
                with (PUB / name).open('a') as stream: stream.write('\n' + NOTE + '\n')
            value = dict(value, CPU_revision_runtime_sha256=sha(REV / 'REVISION_RUNTIME.json'),
                original_table_readback_sha256=sha(legacy.REV / 'ORIGINAL_TABLE_READBACK.json'),
                output_files={str(p.relative_to(ROOT)): sha(p) for p in PUB.rglob('*') if p.is_file()})
        initial_write(path, value)
    original.write = annotated_first_write
    try:
        original.run()
    finally:
        original.write = initial_write
    write(REV / 'COMPLETION.json', dict(status='CPU_projection_and_plots_complete_pending_actual_root',
        CPU_only=True, actual_new_model_calls=0,
        CPU_completion_sha256=sha(BASE / 'CPU_COMPLETION.json'),
        revision_runtime_sha256=sha(REV / 'REVISION_RUNTIME.json'),
        original_GPU_runtime_unchanged=True, time=time.time()))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare()
        print('CPU_PROJECTION_REVISION003_LOCKED', sha(REV / 'REVISION_RUNTIME.json'), flush=True)
    else:
        run()
