"""Join original diagnostic FINAL and complete original LN-only FINAL exactly."""
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha, status
from scripts.close_stvg_opd_p7_phase_v1 import archive
from scripts.publish_stvg_opd_actionability_audit_v1 import git, CHECKOUT, BRANCH, put, safe, receipt_metadata
BASE = ROOT/'artifacts/stvg_opd_mechanism_review_v1'
LN = BASE/'cross_parent_LN'
OUT = ROOT/'results/stvg_opd_mechanism_review/2026-10-13/complete_scope_closing'


def close():
    from scripts.stvg_opd_mechanism_LN_order_common_v1 import verify
    verify()
    diagnostic = read(BASE/'DIAGNOSTIC_FINAL_COMPLETION.json')
    oldview = read(BASE/'ROOT_VISUAL_REVIEW.json')
    oldpublic = read(BASE/'DIAGNOSTIC_PUBLICATION_COMPLETION_RECEIPT.json')
    oldremote = read(BASE/'FINITE_CLOSING_GITHUB_RECEIPT.json')
    assert diagnostic['status'] == 'complete' and oldview['status'] == 'pass'
    assert diagnostic['root_closing_sha256'] == sha(BASE/'ROOT_DIAGNOSTIC_CLOSING_RECEIPT.json')
    assert diagnostic['three_actual_PDF_views_complete'] and oldview['PDF_pages'] == 3
    assert oldpublic['status'] == 'complete' and oldpublic['three_batches_actual_remote_verified']
    assert oldremote['status'] == 'pass' and oldremote['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    assert oldpublic['publication_receipts']['FINITE_CLOSING_GITHUB_RECEIPT.json'] == sha(BASE/'FINITE_CLOSING_GITHUB_RECEIPT.json')
    final = read(LN/'FINAL_COMPLETION.json')
    view = read(LN/'ROOT_VISUAL_REVIEW.json')
    assert final['status'] == 'complete' and view['status'] == 'pass' and view['actual_pixel_view']
    assert all(final[k] for k in ['actual_root', 'actual_PDF_pixel_view', 'actual_remote_publication', 'actual_archive'])
    assert read(LN/'CLOSING_FINAL_GITHUB_RECEIPT.json')['status'] == 'pass'
    assert read(LN/'ARCHIVE_POST_CLOSE_SNAPSHOT_RECEIPT.json')['status'] == 'pass'
    views = {}
    for relative, digest in oldview['files'].items():
        p = ROOT/relative
        assert sha(p) == digest
        views[relative] = dict(sha256=digest, bytes=p.stat().st_size, pages=1, prior_actual_pixel_view_reused=True)
    for relative, b in view['PDF_files'].items():
        p = ROOT/relative
        assert sha(p) == b['sha256'] and p.stat().st_size == b['bytes']
        views[relative] = dict(b, pages=1, newly_actually_viewed=True)
    assert len(views) == 5
    bindings = {}
    for n in ['DIAGNOSTIC_FINAL_COMPLETION.json', 'ROOT_DIAGNOSTIC_CLOSING_RECEIPT.json', 'ROOT_VISUAL_REVIEW.json',
              'DIAGNOSTIC_PUBLICATION_COMPLETION_RECEIPT.json', 'FINITE_CLOSING_GITHUB_RECEIPT.json', 'ARCHIVE_POST_CLOSE_SNAPSHOT_RECEIPT.json']:
        bindings[n] = sha(BASE/n)
    for n in ['ROOT_COMPLETE_READBACK.json', 'ROOT_VISUAL_REVIEW.json', 'ROOT_CLOSING.json', 'FINAL_COMPLETION.json',
              'CLOSING_FINAL_GITHUB_RECEIPT.json', 'ARCHIVE_POST_CLOSE_SNAPSHOT_RECEIPT.json']:
        bindings['cross_parent_LN/'+n] = sha(LN/n)
    scope = 'complete all original N requests: preserved finite CPU mechanism diagnosis and actual768 missing LN-only fits/2304 old controls/three actual orders/source/math/state/input/dense/bootstrap/root/five actual numerical PDF pages/all anonymous outcomes/public/archive; cost32 and future decisions remain separate'
    write(BASE/'ROOT_ALL_AUTHORIZED_N_SCOPES_READBACK.json', dict(status='pass', scope=scope,
        bindings=bindings, CPU_diagnosis_complete=True, cross_parent_LN_complete=True,
        new_formal_LN_fits=768, reused_control_fits=2304, original_qualification_GPU_fits=24,
        actual_PDFs=5, actual_PDF_pages=5, new_GPU_or_GT_scoring_calls_in_join=0,
        remaining_cost32_complete=False, CPU_only=True, GT_read=False, time=time.time()))
    write(BASE/'ROOT_ALL_AUTHORIZED_N_VISUAL_REVIEW.json', dict(status='pass', scope='exact reuse of three prior actually viewed diagnostic PDFs and two newly actually viewed LN PDFs; original diagnostic view receipt remains unchanged',
        actual_pixel_view=True, PDF_files=views, PDFs=5, pages=5, new_actual_pages_viewed=2,
        prior_actual_pages_viewed_reused=3, all_bindings_exact=True, CPU_only=True, GT_read=False, time=time.time()))
    note = '**2026-10-13｜N原有限CPU诊断与缺失LN-only三顺序两项全部真正FINAL。** 原诊断1792arrival/3584既有fit/每轮和按序独立root、原三PDF三页真实view及匿名全部正负public/archive沿实际SHA/旧真正FINAL复用；新原768 LN-only/2304controls三真实order/source-origin/math-state-input-dense/完整资格对照及独立bootstrap/root、新两PDF两页真实view/全部3072匿名正负neutralunknown/256配对public/archive均实际关闭，合五PDF五页。保持原ROOT_VISUAL_REVIEW诊断收据不改，另存ROOT_ALL_AUTHORIZED_N_VISUAL_REVIEW汇总真实view。N scope完整不是原32audited cost/后续主实验全部完成；取消剩余baseline13747同PID/optimizer/ordinarylease T standby不恢复，旧CPU计数/根数学reader分派已保存解释，不重跑GPU/model/DINO/原GT评分、不新gate/因果state方法/论文/CURRENT晋升。\n\n'
    p = ROOT/'docs/RESEARCH_HISTORY.md'
    text = p.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n', '### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor+note, 1)
    p.write_text(text)
    archive(BASE/'archive_all_N_actual_closing', 'ARCHIVE_ALL_N_CLOSING_RECEIPT.json')
    write(BASE/'ROOT_CLOSING.json', dict(status='complete', scope=scope, bindings=bindings,
        root_all_scopes_sha256=sha(BASE/'ROOT_ALL_AUTHORIZED_N_SCOPES_READBACK.json'),
        view_all_scopes_sha256=sha(BASE/'ROOT_ALL_AUTHORIZED_N_VISUAL_REVIEW.json'),
        archive_sha256=sha(BASE/'ARCHIVE_ALL_N_CLOSING_RECEIPT.json'),
        actual_root=True, actual_PDF_pixel_view=True, actual_remote_publication=True, actual_archive=True,
        PDFs=5, pages=5, CPU_only=True, GT_read=False, time=time.time()))
    write(BASE/'FINAL_COMPLETION.json', dict(status='complete', scope=scope, root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        original_two_N_items_complete=True, CPU_diagnosis_complete=True, cross_parent_LN_complete=True,
        actual_root=True, actual_PDF_pixel_view=True, actual_remote_publication=True, actual_archive=True,
        cost32_complete=False, whole_followup_request_complete=False, baseline_remaining_formal_cancelled=True,
        new_gate_created=False, new_state_intervention_launched=False, production_promoted=False,
        manuscript_written=False, CPU_only=True, GT_read=False, time=time.time()))
    status(BASE/'STATUS.json', dict(status='complete_all_original_authorized_N_scopes_cost32_remaining',
        N_complete=True, whole_followup_request_complete=False, CPU_only=True, GT_read=False, time=time.time()))
    print('N_ALL_ORIGINAL_SCOPES_ACTUAL_COMPLETE')


def stage():
    assert read(BASE/'FINAL_COMPLETION.json')['status'] == 'complete'
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    base = git('rev-parse', 'HEAD')
    assert base == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/'+BRANCH)
    assert not git('diff', 'HEAD', '--name-only') and not git('diff', '--cached', '--name-only')
    unrelated = git('ls-files', '--others', '--exclude-standard').splitlines()
    owned = set()
    def copy(p, relative):
        if p.suffix == '.json':
            safe(read(p))
        else:
            ast.parse(p.read_text())
        dest = CHECKOUT/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        assert sha(p) == sha(dest)
        owned.add(relative)
    copy(Path(__file__).resolve(), 'scripts/close_stvg_opd_mechanism_scopes_v1.py')
    for n in ['ROOT_ALL_AUTHORIZED_N_SCOPES_READBACK.json', 'ROOT_ALL_AUTHORIZED_N_VISUAL_REVIEW.json',
              'ROOT_CLOSING.json', 'FINAL_COMPLETION.json', 'ARCHIVE_ALL_N_CLOSING_RECEIPT.json',
              'DIAGNOSTIC_PUBLICATION_COMPLETION_RECEIPT.json']:
        p = BASE/n
        put(OUT/n, dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
        copy(OUT/n, str((OUT/n).relative_to(ROOT)))
    p = LN/'CLOSING_FINAL_GITHUB_RECEIPT.json'
    put(OUT/'cross_parent_LN/CLOSING_FINAL_GITHUB_RECEIPT.json', dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
    copy(OUT/'cross_parent_LN/CLOSING_FINAL_GITHUB_RECEIPT.json', str((OUT/'cross_parent_LN/CLOSING_FINAL_GITHUB_RECEIPT.json').relative_to(ROOT)))
    manifest = OUT/'SOURCE_MANIFEST.json'
    write(manifest, dict(status='pass', scope='exact joining of original finite N diagnosis and original768 LN actual FINAL; all complete outcomes separately bound in immutable verified commits; excludes self hash',
        files={n: dict(sha256=sha(CHECKOUT/n), bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned)}))
    copy(manifest, str(manifest.relative_to(ROOT)))
    entries = []
    for n in sorted(owned):
        p = CHECKOUT/n; data = p.read_bytes()
        entries.append(dict(path=n, bytes=len(data), sha256=sha(p), blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='whole-N-index-') as folder:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(folder)/'index')}
        git('read-tree', base, env=env)
        for e in entries:
            h = git('hash-object', '-w', str(CHECKOUT/e['path'])); assert h == e['blob_sha']
            git('update-index', '--add', '--cacheinfo', '100644', h, e['path'], env=env)
        tree = git('write-tree', env=env)
    write(BASE/'CLOSING_PUBLIC_STAGE.json', dict(status='reviewed_staged', repository='Zonglin-He/A', branch=BRANCH,
        base_commit=base, expected_tree=tree, files=entries, file_count=len(entries), bytes=sum(e['bytes'] for e in entries),
        unrelated_untracked_preserved=unrelated, scope='complete original two N scopes joining FINAL; not audited cost32 or all future work', time=time.time()))
    print('N_ALL_SCOPES_CLOSING_PUBLIC_STAGE', len(entries), tree)


if __name__ == '__main__':
    stage() if len(sys.argv) > 1 and sys.argv[1] == 'stage' else close()
