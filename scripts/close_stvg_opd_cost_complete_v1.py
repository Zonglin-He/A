"""Genuine original32 audited-cost closure; cancelled baseline never resumes."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha, status
from scripts.run_stvg_opd_cost_after_N_priority001 import SCIENCE as BASE, verify
from scripts.close_stvg_opd_p7_phase_v1 import archive


def run():
    verify()
    for n in ['ROOT_COMPLETE_TIMED_FIT_READBACK.json', 'ROOT_FRESH_EXPERT_AND_TIMING_READBACK.json',
              'ROOT_INDEPENDENT_TIMING_STATISTICS.json', 'PUBLIC_SCALAR_AUDIT.json']:
        assert read(BASE/n)['status'] == 'pass'
    view = read(BASE/'ROOT_VISUAL_REVIEW.json')
    public = read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert view['status'] == public['status'] == 'pass' and view['actual_pixel_view']
    assert public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    for relative, b in view['PDF_files'].items():
        assert sha(ROOT/relative) == b['sha256'] and (ROOT/relative).stat().st_size == b['bytes']
    note = f"**2026-10-13｜原32 audited cost实际完整有限FINAL。** 原每域16固定历史query/32真正同步Full路径、32完整科学fit/native/source/pixel/frame/原fresh expert/scientific bitwise和800完整轮记录（实际680 backward）、全部独立五阶段sum/median/p95与无信息计数/root、一PDF一页真实pixel view/全部匿名slow/no-information timings/public/archive实际齐全；{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际pass。HC total median/p95=5.1417/6.8284秒、Vid=3.6825/4.2603秒，实际DINO64＋48=112，Vid4路径0DINO/0admitted/0backward、HC2路径0backward均完整保留。各路径五阶段精确加和，total分位不能用分阶段分位相加；source权重resident/每query空application frame cache而OS pages未强制清空/已有parses/同步每replay和工程recorder/原S暂停CUDAcontext仍resident，O012及inactive专家CPU存储和S身份检查均实计native-prefix，CPU hashes/math/serialization在timer外。是原历史实现审计成本，不是生产throughput或全GPU显存，不能与C不同名单/仪表的128 paired paths混成同一benchmark。GT_read=false，原科学source/runtime/list/frame/precision/optimizer/teacher/随机key/timer不改，只另存C/N真正FINAL后调度；未恢复已取消baseline/旧idleT waiters，未写论文/新gate或晋升CURRENT。仅cost有限scope关闭，S既有orders和取消handoff整体root/public整理与后续决定仍需具体核定。\n\n"
    ledger = ROOT/'docs/RESEARCH_HISTORY.md'; text = ledger.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n', '### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(anchor) == 1; text = text.replace(anchor, anchor+note, 1)
    ledger.write_text(text)
    archive(BASE/'archive_cost_complete_closing', 'ARCHIVE_VERIFICATION_RECEIPT.json')
    scope = 'complete original32 fixed audited Full paths/source-native-expert-scientific math/independent stage sums and descriptive median-p95/all anonymous slow-no-information outcomes/actual PDF pixel view/public/archive; not missing cancelled formal baseline results or whole future followup'
    bindings = {n: sha(BASE/n) for n in ['ROOT_COMPLETE_TIMED_FIT_READBACK.json', 'ROOT_FRESH_EXPERT_AND_TIMING_READBACK.json',
        'ROOT_INDEPENDENT_TIMING_STATISTICS.json', 'ROOT_VISUAL_REVIEW.json', 'PUBLIC_SCALAR_AUDIT.json', 'FINAL_GITHUB_RECEIPT.json', 'ARCHIVE_VERIFICATION_RECEIPT.json']}
    write(BASE/'ROOT_CLOSING.json', dict(status='complete', scope=scope, bindings=bindings, actual_timed_paths=32,
        actual_root=True, actual_PDF_pixel_view=True, actual_remote_publication=True, actual_archive=True,
        CPU_only=True, GT_read=False, time=time.time()))
    write(BASE/'FINAL_COMPLETION.json', dict(status='complete', scope=scope, root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        actual_timed_paths=32, actual_root=True, actual_PDF_pixel_view=True, actual_remote_publication=True, actual_archive=True,
        all_slow_and_no_information_paths_retained=True, baseline_remaining_formal_cancelled=True,
        whole_followup_request_complete=False, production_promoted=False, manuscript_written=False, CPU_only=True, GT_read=False, time=time.time()))
    state = dict(status='complete_original32_audited_cost_scope', actual_timed_paths=32, CPU_only=True, GT_read=False,
                 baseline_remaining_formal_cancelled=True, whole_followup_request_complete=False, time=time.time())
    status(BASE/'STATUS.json', state)
    status(BASE/'after_N_priority001/STATUS.json', state)
    print('COST32_ACTUAL_FINITE_SCOPE_CLOSED', public['commit'])


if __name__ == '__main__':
    run()
