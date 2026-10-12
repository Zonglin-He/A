"""Close only after all original LN root/view/anonymous remote/archive evidence."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha
from scripts.stvg_opd_mechanism_LN_order_common_v1 import BASE, verify
from scripts.close_stvg_opd_p7_phase_v1 import archive


def run():
    verify()
    root = read(BASE/'ROOT_COMPLETE_READBACK.json')
    view = read(BASE/'ROOT_VISUAL_REVIEW.json')
    public = read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert root['status'] == view['status'] == public['status'] == 'pass'
    assert root['new_formal_fits'] == 768 and root['reused_control_fits'] == 2304
    assert view['actual_pixel_view'] and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    assert read(BASE/'PUBLIC_SCALAR_AUDIT.json')['status'] == 'pass'
    for rel, binding in view['PDF_files'].items():
        assert sha(ROOT/rel) == binding['sha256'] and (ROOT/rel).stat().st_size == binding['bytes']
    scope = 'complete original historical128 parents per target/three actual orders/all768 new LN-only and2304 exact controls/all source/math/state/input/dense/independent parent bootstrap/actual two PDF pixel views/all anonymous outcomes/public/archive; cost32 remains separate'
    note = f"**2026-10-13｜原LN-only768与四臂三顺序完整有限scope实际FINAL。** 两域各128历史parent/三真实order，全768新LN-only及原2304 controls精确复用；1536active LN/256query frozen/1792stored state、0新DINO，全global后GT。完整3072fit/source-origin/reset-inheritance-writeback/input/math/dense、48qualified-formal（新12/旧36）及新12完整repeat、独立parent平均后10000bootstrap真实pass；两PDF两页实际pixel view/3072匿名全正负neutralunknown/256配对和全部current严重负尾完整公开。{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际pass。Full−LN-only pooled HC−0.169pp/CI[-0.679,+0.246]跨零，Vid+0.464/[+0.098,+0.922]正；仅支持原历史域的整体在线组件差异，不是逐query LN因果/gate或普遍必要性，不与C205/700两order跨池合并。旧末尾36计数应为单臂12、旧控制P4/newLN P3根分派均保存原件并独立接续，0重跑GPU/新GT评分，science/runtime不改。原32audited cost和后续决定仍须实际串行，剩余formal baseline维持人类cancel13747同PID/optimizer/lease T standby，未写论文或晋升CURRENT。见[完整结果](</home/wwww/visual grounding/results/stvg_opd_mechanism_review/2026-10-13/LN_complete/REPORT.md>)。\n\n"
    ledger = ROOT/'docs/RESEARCH_HISTORY.md'
    text = ledger.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n', '### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor+note, 1)
    ledger.write_text(text)
    archive(BASE/'archive_LN_complete_closing', 'ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings = {n: sha(BASE/n) for n in ['ROOT_COMPLETE_READBACK.json', 'ROOT_ANONYMOUS_SCALAR_READBACK.json',
        'ROOT_VISUAL_REVIEW.json', 'PUBLIC_SCALAR_AUDIT.json', 'FINAL_GITHUB_RECEIPT.json', 'ARCHIVE_VERIFICATION_RECEIPT.json']}
    write(BASE/'ROOT_CLOSING.json', dict(status='complete', scope=scope, bindings=bindings,
        actual_root=True, actual_PDF_pixel_view=True, actual_remote_publication=True, actual_archive=True,
        CPU_only=True, GT_read=True, original_source_and_runtime_unchanged=True, time=time.time()))
    write(BASE/'FINAL_COMPLETION.json', dict(status='complete', scope=scope, root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        actual_root=True, actual_PDF_pixel_view=True, actual_remote_publication=True, actual_archive=True,
        new_formal_fits=768, reused_control_fits=2304, all3072_anonymous_outcomes_retained=True,
        new_DINO_calls=0, production_promoted=False, new_gate_created=False, manuscript_written=False,
        cost32_complete=False, whole_followup_request_complete=False, baseline_remaining_formal_cancelled=True,
        CPU_only=True, GT_read=True, time=time.time()))
    state = dict(status='complete_original_LN_scope_cost32_remaining', scope=scope,
        new_formal_fits=768, reused_control_fits=2304, CPU_only=True, GT_read=True,
        latest_CPU_revision='CPU_count_dispatch001', whole_followup_request_complete=False, time=time.time())
    status(BASE/'STATUS.json', state)
    status(BASE/'after_C_priority001/STATUS.json', state)
    print('LN_ACTUAL_FINITE_SCOPE_CLOSED', public['commit'])


if __name__ == '__main__':
    run()
