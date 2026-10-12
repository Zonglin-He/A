"""Preserve the completed 768 LN scores; repair only their final count dispatch."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import shutil
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha
from scripts.stvg_opd_mechanism_LN_order_common_v1 import BASE, PUB, stages, verify, ARMS
REV = BASE / 'CPU_count_dispatch001'
ORIGINAL = ROOT / 'scripts/score_stvg_opd_mechanism_LN_order_v1.py'


def run():
    verify()
    assert ARMS == ['LN_only']
    barrier = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert barrier['status'] == 'sealed' and barrier['new_formal_fits'] == 768
    expected = sum(len(s['conditions']) * len(s['orders']) * len(ARMS) * 2 for s in stages().values())
    assert expected == 12
    preserved = [ORIGINAL, BASE / 'RUNTIME_LOCK.json', BASE / 'DESIGN_LOCK.json',
                 BASE / 'GLOBAL_PREDICTION_BARRIER.json', BASE / 'GT_EXPOSURE.json',
                 BASE / 'after_C_priority001/STATUS.json', BASE / 'STATUS.json',
                 BASE / 'after_C_priority001/logs/score_stvg_opd_mechanism_LN_order_v1.log',
                 BASE / 'after_C_priority001/logs/controller.log']
    count = first = 0
    totals = {}
    immutable = []
    for name, stage in stages().items():
        cpu = BASE / 'stages' / name / 'CPU_COMPLETION.json'
        rows = PUB / name / 'ROWS.jsonl.gz'
        c = read(cpu)
        assert c['status'] == 'postseal_complete_math_state_input_dense'
        assert sha(rows) == c['rows_sha256']
        count += c['counts']['complete_fit_math']
        first += c['counts']['qualified_formal_pairs']
        for k, v in c['counts'].items():
            totals[k] = totals.get(k, 0) + v
        preserved.extend([cpu, rows, BASE / 'stages' / name / 'PREDICTION_BARRIER.json'])
        immutable.extend([cpu, rows])
        for condition in stage['conditions']:
            for order in stage['orders']:
                for at in range(2):
                    f = BASE / 'first_formal' / name / condition / order / 'LN_only' / f'{at:05}.json'
                    assert read(f)['status'] == 'pass'
                    preserved.append(f)
    assert count == 768 and first == expected
    runtime_sha = sha(BASE / 'RUNTIME_LOCK.json')
    before = {str(p.relative_to(ROOT)): sha(p) for p in immutable}
    REV.mkdir(parents=True, exist_ok=False)
    originals = {}
    for p in preserved:
        target = REV / 'originals' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        assert sha(target) == sha(p)
        originals[str(p.relative_to(ROOT))] = dict(sha256=sha(p), bytes=p.stat().st_size)
    write(REV / 'CAPTURE_RECEIPT.json', dict(status='preserved',
        scope='terminal scorer assertion expected three arms although only LN_only is present; all stage scores and scientific pins preserved',
        accepted_formal_fits=768, original_expected_first_formal_pairs=36,
        actual_and_locked_first_formal_pairs=expected, preserved_files=originals,
        CPU_only=True, GT_read=True, time=time.time()))
    text = ORIGINAL.read_text()
    old = "assert counts['complete_fit_math'] == 768 and counts['qualified_formal_pairs'] == 36"
    new = "assert counts['complete_fit_math'] == 768 and counts['qualified_formal_pairs'] == 12"
    assert text.count(old) == 1
    write(REV / 'REVISION_RUNTIME.json', dict(status='locked_CPU_dispatch_only',
        original_source_sha256=sha(ORIGINAL), original_runtime_sha256=runtime_sha,
        dispatcher_sha256=sha(Path(__file__)), changes=[dict(original=old, replacement=new)],
        expectation_derived_from_locked_scope=True, all_saved_scores_reused_exactly=True,
        GPU_calls=0, new_model_calls=0, new_DINO_calls=0, new_GT_scoring_calls=0,
        original_scientific_source_and_runtime_unchanged=True, CPU_only=True, time=time.time()))
    ns = {'__file__': str(ORIGINAL), '__name__': 'cpu_count_dispatch001'}
    exec(compile(text.replace(old, new), str(ORIGINAL), 'exec'), ns)
    ns['run']()
    assert all(sha(ROOT / p) == h for p, h in before.items())
    complete = read(BASE / 'CPU_SCORING_COMPLETION.json')
    for k, v in totals.items():
        assert complete['counts'][k] == v
    from scripts.finalize_stvg_opd_mechanism_LN_order_v1 import run as finalize
    finalize()
    verify()
    assert sha(BASE / 'RUNTIME_LOCK.json') == runtime_sha
    import torch
    assert not torch.cuda.is_initialized()
    write(REV / 'RESOLUTION_RECEIPT.json', dict(status='pass',
        scope='same original 768 scored rows and 12 accepted qualified-formal pairs; repaired final count receipt and resumed original scalar statistics',
        counts=complete['counts'], all_preserved_rows_and_stage_receipts_unchanged=True,
        source_sha256=sha(ORIGINAL), runtime_sha256=runtime_sha,
        CPU_scoring_completion_sha256=sha(BASE / 'CPU_SCORING_COMPLETION.json'),
        CPU_completion_sha256=sha(BASE / 'CPU_COMPLETION.json'),
        original_scientific_source_and_runtime_unchanged=True,
        GPU_calls=0, model_calls=0, DINO_calls=0, new_GT_scoring_calls=0,
        CUDA_initialized=False, CPU_only=True, GT_read=True, time=time.time()))
    state = dict(status='pending_actual_LN_complete_independent_root_PDF_pixel_view_publication_archive_then_cost',
        scope='all768 LN-only fits/global/stage scores and old scalar statistics completed; actual root/view/public/archive remain',
        new_formal_fits=768, reused_formal_fits=2304, CPU_only=True, GT_read=True,
        experiment_complete=False, latest_CPU_revision='CPU_count_dispatch001', time=time.time())
    status(BASE / 'STATUS.json', state)
    status(BASE / 'after_C_priority001/STATUS.json', state)
    write(REV / 'CPU_COMPLETION.json', state)
    ledger = ROOT / 'docs/RESEARCH_HISTORY.md'
    note = '**2026-10-13｜N全部768 LN-only正式已封存，末尾CPU计数收据另存接续。** 两域各384/三固定order全部global后原GT，原768完整math/state/input/dense与全部评分输出已保存；原单臂scorer末尾误要求36 qualified-formal，而原2域×3order×每序2first×1 LN-only臂实际12、逐receipt均pass。另存CPU_count_dispatch001保存原source/runtime/全部rows/旧失败log和状态，只修汇总期望12、精确复用评分后接续原四臂统计，0新GPU/model/DINO/GT评分；原科学锁与全部已接受预测不改。此为已解释CPU末尾dispatch错误，不当GPU/科学fit故障；完整独立root/PDF真实view/匿名全正负public/archive/真正N FINAL仍须根实际继续，原32audited cost与后续决定未齐，取消baseline13747同原PID/optimizer/lease T standby不恢复。见[保存与修正收据](</home/wwww/visual grounding/artifacts/stvg_opd_mechanism_review_v1/cross_parent_LN/CPU_count_dispatch001/RESOLUTION_RECEIPT.json>)。'
    content = ledger.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n', '### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert content.count(anchor) == 1
        content = content.replace(anchor, anchor + note + '\n\n', 1)
    ledger.write_text(content)
    from scripts.close_stvg_opd_p7_phase_v1 import archive
    archive(REV / 'archive_resolution001', 'ARCHIVE_RESOLUTION_RECEIPT.json')
    print('LN_CPU_COUNT_DISPATCH_RESOLVED', count, expected, flush=True)


if __name__ == '__main__':
    run()
