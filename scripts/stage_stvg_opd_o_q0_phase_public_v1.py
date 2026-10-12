"""Explicit anonymous science/source/receipt export for completed O and Native Q0.

Private inputs, model weights, raw fits, predictions, annotations, media, dense
Jacobians and vector-valued witnesses remain outside this finite whitelist.
The original S publisher and all experiment science/runtime files are untouched.
"""
import ast
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.publish_stvg_opd_supplement_v1 import safe, private_attachment_path

O = ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
Q = ROOT/'artifacts/stvg_opd_query_observability_q0_v1/Native_support_scope003'
OP = ROOT/'results/stvg_opd_optimization_geometry/2026-10-12/common_state_public'
QP = ROOT/'results/stvg_opd_query_observability_q0/2026-10-12/Native_support_scope003_public'
CHECKOUT = ROOT.parent/'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def scalar_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def gzip_rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in data:
                safe(row)
                stream.write((json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n').encode())


def receipt_metadata(value):
    """Only count/integrity/boolean/state metadata, with no numerical witnesses."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, child in value.items():
        if key in {'records', 'files', 'originals', 'accepted_prefix', 'prefix', 'settings',
                   'pins_original', 'original_baseline_processes', 'protected',
                   'target_thread_id', 'targetThreadId', 'automation_id',
                   'unique_monitor_agent', 'query_ordinal', 'failed_query_ordinal',
                   'qualification_indices', 'private_source_paths', 'command', 'evidence'}:
            continue
        if key == 'pins' and isinstance(child, dict):
            result[key] = {path: digest for path, digest in child.items()
                           if isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                           and not private_attachment_path(path)}
        elif key in {'counts', 'checks'} and isinstance(child, dict):
            result[key] = receipt_metadata(child)
        elif isinstance(child, bool):
            result[key] = child
        elif isinstance(child, str) and (key in {'status', 'scope', 'stage', 'dataset', 'source'}
                                       or re.fullmatch('[0-9a-f]{40,64}', child)):
            result[key] = child.replace(str(ROOT), '<PROJECT_ROOT>')
        elif isinstance(child, (int, float)) and re.search(
                'count|total|done|cells|fits|rounds|scalar_checks|coordinates|pairs|hashes|bytes|elapsed|time|pid|PDF|queries|decodes', key, re.I):
            result[key] = child
    safe(result)
    return result


def run():
    assert read(O/'GLOBAL_PREDICTION_BARRIER.json')['logical_fits'] == 5376
    assert read(O/'postseal_independent_root001/ROOT_READBACK.json')['status'] == 'pass'
    assert read(O/'postseal_independent_root001/ROOT_VISUAL_REVIEW.json')['status'] == 'pass'
    assert read(Q/'GLOBAL_PREDICTION_BARRIER.json')['cells'] == 64
    assert read(Q/'ROOT_STATISTICS_READBACK.json')['status'] == 'pass'
    assert read(Q/'ROOT_VISUAL_REVIEW.json')['status'] == 'pass'
    for folder in (O/'publication_phase001', Q/'publication_phase001', OP, QP):
        folder.mkdir(parents=True, exist_ok=False)

    def git(*args, env=None):
        return subprocess.check_output(['git', *args], cwd=CHECKOUT, text=True, env=env).strip()
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    base = git('rev-parse', 'HEAD')
    assert base == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/'+BRANCH)
    assert not git('diff', 'HEAD', '--name-only') and not git('diff', '--cached', '--name-only')
    unrelated = git('ls-files', '--others', '--exclude-standard').splitlines()
    owned = set()
    bindings = {}

    def copy(original, relative):
        destination = CHECKOUT/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        assert sha(destination) == sha(original)
        owned.add(relative)

    old_o = ROOT/'results/stvg_opd_optimization_geometry/2026-10-12/common_state'
    old_q = ROOT/'results/stvg_opd_query_observability_q0/2026-10-12/Native_support_scope003'
    parent_o = {}
    data = scalar_rows(old_o/'ROWS.jsonl.gz')
    projected = []
    for index, row in enumerate(data):
        key = (row['dataset'], row['source_id'])
        parent_o.setdefault(key, len(parent_o))
        public = {k: v for k, v in row.items() if k not in {'source_id', 'query_ordinal'}}
        public.update(parent_id=parent_o[key], public_cell=index)
        assert {k: v for k, v in public.items() if k not in {'parent_id', 'public_cell'}} == {
            k: v for k, v in row.items() if k not in {'source_id', 'query_ordinal'}}
        projected.append(public)
    assert len(projected) == 1792 and len(parent_o) == 256
    gzip_rows(OP/'ROWS.jsonl.gz', projected)
    parent_q = {}
    query_q = {}
    for row in scalar_rows(old_q/'ROWS.jsonl.gz'):
        key = (row['dataset'], row['development_parent'])
        parent_q.setdefault(key, len(parent_q))
        query_q.setdefault((row['dataset'], row['query_ordinal']), len(query_q))
    for name in ('ROWS.jsonl.gz', 'CANDIDATE_ROWS.jsonl.gz'):
        public_rows = []
        for row in scalar_rows(old_q/name):
            public = {k: v for k, v in row.items() if k not in {'development_parent', 'query_ordinal'}}
            public.update(parent_id=parent_q[row['dataset'], row['development_parent']],
                          public_query=query_q[row['dataset'], row['query_ordinal']])
            assert {k: v for k, v in public.items() if k not in {'parent_id', 'public_query'}} == {
                k: v for k, v in row.items() if k not in {'development_parent', 'query_ordinal'}}
            public_rows.append(public)
        gzip_rows(QP/name, public_rows)
    assert len(parent_q) == len(query_q) == 64
    for old, public, names in [(old_o, OP, ['SUMMARY.json', 'INDEPENDENT_ROOT_STATISTICS.json']),
                               (old_q, QP, ['SUMMARY.json'])]:
        for name in names:
            safe(read(old/name));shutil.copy2(old/name, public/name)
        text = (old/'REPORT.md').read_text()
        if public == OP:
            text = text.replace('The PDF figures still require actual pixel view; publication, remote content verification, archive closure and the true common-state FINAL remain outstanding.',
                'Both PDF figures passed actual root pixel view. Publication and archive closing are bound separately by the genuine phase-closing receipts; this phase does not close S/C/N/Q0.')
        else:
            text = text.replace('Two PDFs still require actual pixel view, anonymized publication/remote content verification, archive and true FINAL.',
                'Both PDFs passed actual root pixel view. Anonymous publication, remote content and archive closing are recorded by the genuine corrected-Native Q0 closing receipts; this phase does not close S/C/N/O.')
        (public/'REPORT.md').write_text(text)
    cohort = read(O/'COMMON_STATE_COHORT.json')['records']
    protocols = []
    for row in cohort:
        value = {k: row[k] for k in ('dataset', 'source', 'stage', 'condition', 'order', 'arrival', 'development', 'setting', 'config')}
        value.update(public_cell=row['index'], parent_id=parent_o[row['dataset'], row['source_id']],
                     original_Full_sha256=row['original_Full']['sha256'],
                     matched_input_sha256=row['original_Full']['input_sha256'],
                     arm_files_sha256={arm:sha(O/'common_state'/f"{row['index']:05}"/(arm+'.pt'))
                                       for arm in ('Full', 'Direct', 'Direct_capped')})
        protocols.append(value)
    gzip_rows(OP/'ANONYMOUS_PROTOCOL_AND_INTEGRITY.jsonl.gz', protocols)
    design = read(Q/'DESIGN_LOCK.json')
    protocols = []
    for row in design['records']:
        value = {k: row[k] for k in ('dataset', 'source', 'order', 'arrival', 'config')}
        value.update(public_query=query_q[row['dataset'], row['query_ordinal']],
                     parent_id=parent_q[row['dataset'], row['development_parent']],
                     original_Full_sha256=row['original_Full']['sha256'],
                     matched_input_sha256=row['input']['sha256'],
                     formal_episode_sha256=sha(Q/'formal'/f"{row['index']:05}.pt"))
        protocols.append(value)
    gzip_rows(QP/'ANONYMOUS_PROTOCOL_AND_INTEGRITY.jsonl.gz', protocols)
    write(QP/'CORRECT_NATIVE_SCOPE.json', dict(status='complete_64_correct_Native_diagnostics',
          pooled_output_RMS_budget=design['pooled_output_RMS_budget'],
          budget_definition=design['budget_definition'], amplitude_max=design['amplitude_max'],
          directions_max=design['directions_max'], groups=design['groups'],
          historically_exposed_development=True, old_full_video_scope_not_eligible=True,
          optimizer_steps=0, writeback=False, new_DINO_calls=0, independent_online_stream=False))
    runtimes = [O/'GPU_RUNTIME.json', O/'postseal_independent_root001/RUNTIME.json',
                O/'postseal_independent_root001/source_origin_stream003/REVISION_RUNTIME.json',
                Q/'RUNTIME.json', Q/'frozen_memory_revision004/REVISION_RUNTIME.json',
                Q/'postseal_actual_root001/receipt_binding005/REVISION_RUNTIME.json']
    sources = {'scripts/audit_stvg_opd_o_q0_public_v1.py', 'scripts/stage_stvg_opd_o_q0_phase_public_v1.py',
               'scripts/score_stvg_opd_common_state_geometry_v1.py', 'scripts/report_stvg_opd_common_state_postseal_v1.py',
               'scripts/review_stvg_opd_common_state_source003.py', 'scripts/continue_stvg_opd_common_state_source003.py',
               'vg_tta/stvg_opd_common_state_geometry_arithmetic004.py',
               'scripts/run_stvg_opd_common_state_geometry_arithmetic004.py',
               'scripts/run_stvg_opd_common_state_geometry_lease006.py'}
    for runtime_path in runtimes:
        for path, digest in read(runtime_path)['pins'].items():
            assert sha(ROOT/path) == digest, path
            if not path.startswith('artifacts/') and Path(path).suffix in {'.py', '.md', '.json'}:
                sources.add(path)
    for relative in sorted(sources):
        original = ROOT/relative
        text = original.read_text()
        if private_attachment_path(text):
            text = re.sub(r'/home/[^/]+/\.codex/attachments/[^\"\'\n]+', '<LOCAL_AUTHORIZATION_ATTACHMENT>', text)
        assert not private_attachment_path(text)
        if original.suffix == '.py':
            ast.parse(text)
        destination = CHECKOUT/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        owned.add(relative)
        bindings[relative] = dict(private_source_sha256=sha(original), public_source_sha256=sha(destination),
                                  local_authorization_path_only_projected=sha(original) != sha(destination))
    receipt_lists = [
        (O, OP, ['GPU_RUNTIME.json', 'ROOT_QUALIFICATION_READBACK.json', 'GLOBAL_PREDICTION_BARRIER.json',
                 'COMMON_STATE_CPU_COMPLETION.json', 'BASELINE_RESUMPTION_RECEIPT.json',
                 'postseal_independent_root001/ROOT_READBACK.json', 'postseal_independent_root001/ROOT_VISUAL_REVIEW.json',
                 'postseal_independent_root001/source_origin_stream003/CAPTURE_RECEIPT.json',
                 'postseal_independent_root001/source_origin_stream003/CPU_CONTRACTS.json',
                 'postseal_independent_root001/source_origin_stream003/REVISION_RUNTIME.json',
                 'postseal_independent_root001/source_origin_stream003/LAUNCH.json']),
        (Q, QP, ['RUNTIME.json', 'SOURCE_SCOPE_READBACK.json', 'ROOT_QUALIFICATION_READBACK.json',
                 'GLOBAL_PREDICTION_BARRIER.json', 'ROOT_DEPLOYMENT_READBACK.json', 'CPU_COMPLETION.json',
                 'ROOT_STATISTICS_READBACK.json', 'ROOT_VISUAL_REVIEW.json',
                 'frozen_memory_revision004/CAPTURE_RECEIPT.json', 'frozen_memory_revision004/DESIGN_LOCK.json',
                 'frozen_memory_revision004/REVISION_RUNTIME.json', 'frozen_memory_revision004/LAUNCH.json',
                 'frozen_memory_revision004/ROOT_MEMORY_QUALIFICATION_READBACK.json',
                 'frozen_memory_revision004/PRIORITY_PAUSE_RECEIPT.json',
                 'frozen_memory_revision004/PRIORITY_RESUMPTION_RECEIPT.json',
                 'postseal_actual_root001/receipt_binding005/CAPTURE_RECEIPT.json',
                 'postseal_actual_root001/receipt_binding005/CPU_CONTRACTS.json',
                 'postseal_actual_root001/receipt_binding005/REVISION_RUNTIME.json',
                 'postseal_actual_root001/receipt_binding005/LAUNCH.json'])]
    for base_path, public, receipts in receipt_lists:
        for relative in receipts:
            original = base_path/relative
            if original.exists():
                write(public/'receipts'/relative, dict(original_private_receipt_sha256=sha(original),
                                                       projection=receipt_metadata(read(original))))
    for public, output, stems in [(OP, ROOT/'output/pdf/stvg_opd_optimization_geometry/common_state', ['paired_effects', 'parameter_geometry']),
                                  (QP, ROOT/'output/pdf/stvg_opd_query_observability_q0', ['query_readouts', 'query_propagation'])]:
        for stem in stems:
            for suffix in ('pdf', 'png'):
                destination = public/'figures'/f'{stem}.{suffix}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output/f'{stem}.{suffix}', destination)
    for public in (OP, QP):
        write(public/'SOURCE_PROJECTION_MANIFEST.json', dict(status='pass', sources=bindings,
              scope='exact pinned science/engineering/root/report sources; local authorization path redacted only',
              private_payloads_or_annotations_exported=False, personal_conversation_exported=False))
        for original in sorted(public.rglob('*')):
            if original.is_file():
                copy(original, str(original.relative_to(ROOT)))
    command = [str(ROOT/'.conda/tubedetr/bin/python'), '-B',
               str(CHECKOUT/'scripts/audit_stvg_opd_o_q0_public_v1.py'), str(CHECKOUT)]
    result = subprocess.run(command, cwd=CHECKOUT, capture_output=True, text=True,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'})
    assert result.returncode == 0, result.stdout+result.stderr
    audit = json.loads(result.stdout);assert audit['status'] == 'pass'
    for base_path, public, kind in [(O, OP, 'O'), (Q, QP, 'Q0')]:
        write(base_path/'publication_phase001/PUBLIC_SCALAR_AUDIT.json', dict(audit[kind],
              actual_public_checkout=True, CPU_only=True, GT_read=False, time=time.time()))
        write(public/'PUBLIC_SCALAR_AUDIT.json', audit[kind])
        copy(public/'PUBLIC_SCALAR_AUDIT.json', str((public/'PUBLIC_SCALAR_AUDIT.json').relative_to(ROOT)))
    entries = []
    for relative in sorted(owned):
        path = CHECKOUT/relative;data = path.read_bytes()
        entries.append(dict(path=relative, bytes=len(data), sha256=sha(path),
                            blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='stvg-o-q0-phase-index-') as folder:
        env = {**os.environ, 'GIT_INDEX_FILE':str(Path(folder)/'index')}
        git('read-tree', base, env=env)
        for row in entries:
            blob = git('hash-object', '-w', str(CHECKOUT/row['path']))
            assert blob == row['blob_sha']
            git('update-index', '--add', '--cacheinfo', '100644', blob, row['path'], env=env)
        tree = git('write-tree', env=env)
    stage = dict(status='reviewed_staged', repository='Zonglin-He/A', branch=BRANCH,
                 base_commit=base, expected_tree=tree, files=entries, file_count=len(entries),
                 bytes=sum(row['bytes'] for row in entries), unrelated_untracked_preserved=unrelated,
                 scope='all actual O common-state and corrected-Native Q0 positive/negative scalar results, exact sources, root evidence and four actual viewed PDFs; S/C/N remain open',
                 GT_payload_exported=False, time=time.time())
    for base_path in (O, Q):
        write(base_path/'publication_phase001/PUBLIC_STAGE.json', stage)
    print(json.dumps({k:v for k,v in stage.items() if k not in {'files', 'unrelated_untracked_preserved'}}, allow_nan=False))


if __name__ == '__main__':
    run()
