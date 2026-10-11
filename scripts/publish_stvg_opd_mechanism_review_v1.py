"""Stage anonymous CPU evidence and queued LN-only code for exact public review.

This does not publish private fits, GT geometry, video/query content or the
user's manuscript. The authenticated connector commits the reviewed tree.
"""
import ast
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1'
PUB=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'
SOURCES=[
 'protocols/stvg_opd_mechanism_review_v1.md',
 'scripts/analyze_stvg_opd_mechanism_review_v1.py',
 'scripts/review_stvg_opd_mechanism_review_v1.py',
 'scripts/render_stvg_opd_mechanism_review_v1.py',
 'scripts/render_stvg_opd_mechanism_review_layout002.py',
 'scripts/verify_stvg_opd_mechanism_DINO_v1.py',
 'scripts/write_stvg_opd_mechanism_review_report_v1.py',
 'scripts/revise_stvg_opd_reviewer_manuscript_v1.py',
 'scripts/stvg_opd_mechanism_LN_order_common_v1.py',
 'scripts/run_stvg_opd_mechanism_LN_order_v1.py',
 'scripts/score_stvg_opd_mechanism_LN_order_v1.py',
 'scripts/finalize_stvg_opd_mechanism_LN_order_v1.py',
 'scripts/continue_stvg_opd_mechanism_LN_order_v1.py',
 'scripts/publish_stvg_opd_mechanism_review_v1.py',
]
RECEIPTS=[
 'AUTHORIZATION_RECEIPT.json','CPU_RUNTIME.json','CPU_LAUNCH.json','CPU_CONTRACTS.json',
 'GT_EXPOSURE.json','CPU_COMPLETION.json','ROOT_STATISTICS_READBACK.json',
 'REWARD_PILOT_DECISION.json','DINO_RECONSTRUCTION_RUNTIME.json','DINO_RECONSTRUCTION_READBACK.json',
 'FIGURE_EXPORT.json','FIGURE_LAYOUT002_EXPORT.json','ROOT_VISUAL_REVIEW.json',
 'MANUSCRIPT_REVISION_RECEIPT.json','MANUSCRIPT_COMPILATION_RECEIPT.json',
 'HOURLY_MONITOR_SCOPE_UPDATE.json','recovery/figure_layout_002/CAPTURE_RECEIPT.json',
 'recovery/public_source_projection_003/CAPTURE_RECEIPT.json',
 'cross_parent_LN/DESIGN_LOCK.json','cross_parent_LN/RUNTIME_LOCK.json',
 'cross_parent_LN/CPU_CONTRACTS.json','cross_parent_LN/CPU_HISTORICAL_SCOPE_READBACK.json',
 'cross_parent_LN/LAUNCH.json',
]


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=CHECKOUT,env=env,text=True).strip()


def put(path,value):
    """Reuse identical preparation output without rewriting immutable evidence."""
    if path.exists():
        assert read(path)==value,str(path)
    else:
        write(path,value)


def project(value):
    if isinstance(value,dict):
        return {k:project(v) for k,v in value.items() if k not in
                {'records','full_payloads','native_head_VJP','math_audit','log','diagnostics',
                 'private_input_bindings','frames','query_text','caption','attachment_path'}}
    if isinstance(value,list):return [project(v) for v in value]
    if isinstance(value,str):return value.replace(str(ROOT),'<PROJECT_ROOT>')
    return value


def run():
    from scripts.analyze_stvg_opd_mechanism_review_v1 import verify
    from scripts.stvg_opd_mechanism_LN_order_common_v1 import verify as verify_LN
    verify();verify_LN()
    pins={**read(BASE/'CPU_RUNTIME.json')['pins'],**read(BASE/'cross_parent_LN/RUNTIME_LOCK.json')['pins'],
          **read(BASE/'DINO_RECONSTRUCTION_RUNTIME.json')['pins']}
    pin_sources={p for p in pins if Path(p).suffix in {'.py','.md'}}
    for f in ['ROOT_STATISTICS_READBACK.json','ROOT_VISUAL_REVIEW.json','DINO_RECONSTRUCTION_READBACK.json']:
        assert read(BASE/f)['status']=='pass'
    assert read(BASE/'REWARD_PILOT_DECISION.json')['status']=='not_triggered'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD')
    assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    unrelated=[p for p in untracked if 'stvg_motivation' in p.lower()]
    for p in set(untracked)-set(unrelated):
        assert p in SOURCES or p in pin_sources or p.startswith(str(PUB.relative_to(ROOT))+'/'),p
    for rel in RECEIPTS:
        p=BASE/rel;value=project(read(p));safe(value)
        put(PUB/'receipts'/rel,dict(original_private_receipt_sha256=sha(p),projection=value))
    # Final scalar-only figures were actually visually checked, unlike private RGB cases.
    for name in ['quality','propagation','oracles']:
        source=ROOT/'output/pdf/stvg_opd_mechanism_review'/f'{name}.pdf'
        shutil.copy2(source,PUB/f'{name}.pdf');assert sha(source)==sha(PUB/f'{name}.pdf')
    # Validate every scalar row before staging. The large compressed frame record is
    # losslessly segmented for bounded transport; no row or float is rounded/dropped.
    counts={}
    for name in ['QUALITY_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz','CELL_ROWS.jsonl.gz']:
        n=0
        with gzip.open(PUB/name,'rt') as f:
            for line in f:
                safe(json.loads(line));n+=1
        counts[name]=n
    assert counts=={'QUALITY_ROWS.jsonl.gz':3553,'FRAME_ROWS.jsonl.gz':413427,'CELL_ROWS.jsonl.gz':1792}
    chunks=PUB/'frame_scalar_chunks';chunks.mkdir(exist_ok=True)
    records=[];data=(PUB/'FRAME_ROWS.jsonl.gz').read_bytes()
    for i,at in enumerate(range(0,len(data),512*1024)):
        p=chunks/f'FRAME_ROWS.jsonl.gz.part{i:03d}';part=data[at:at+512*1024]
        if p.exists():assert p.read_bytes()==part
        else:p.write_bytes(part)
        records.append(dict(path=str(p.relative_to(PUB)),bytes=len(part),sha256=sha(p)))
    assert b''.join((PUB/r['path']).read_bytes() for r in records)==data
    put(PUB/'FRAME_SCALAR_MANIFEST.json',dict(status='lossless_segments_of_original_anonymous_gzip',
          original_file='FRAME_ROWS.jsonl.gz',original_bytes=len(data),original_sha256=sha(PUB/'FRAME_ROWS.jsonl.gz'),
          rows=counts['FRAME_ROWS.jsonl.gz'],concatenate_in_list_order_before_gzip_decode=True,
          no_scalar_rounding_or_record_omission=True,chunks=records))
    put(PUB/'PUBLIC_STAGE_STATUS.json',dict(status='CPU_diagnosis_independent_root_and_actual_figure_view_complete_LN_GPU_queued',
          CPU_diagnostic_complete=True,CPU_matched_cells=1792,CPU_expert_frame_cells=3553,CPU_event_frame_cells=413427,
          root_scalar_comparisons=1682153,new_diagnostic_model_calls=0,
          reward_pilot_trigger='not_triggered',reward_pilot_GPU_launched=False,primary_method_changed=False,
          three_actual_existing_permutation_fits=2304,missing_LN_new_formal_fits_planned=768,
          missing_LN_GPU_qualification_fits_planned=24,missing_LN_new_formal_fits_completed=0,
          missing_LN_actual_GPU_qualification_complete=False,new_LN_GT_read=False,
          expanded_905_parent_state_and_uncached_latency_are_separate_pending_studies=True,
          reviewer_experiment_request_complete=False))
    (PUB/'README.md').write_text(
      '# Reviewer mechanism evidence\n\n'
      'Read REPORT.md and the fixed protocol before the scalar tables. This completed CPU readback '
      'uses already sealed Full/Direct P2 evidence, independently checked aggregation, exact DINO '
      'reconstruction, and three actually viewed scalar PDFs. All positive/negative conditions '
      'and common-parent intersections are retained. It does not change the primary method.\n\n'
      'QUALITY_ROWS and CELL_ROWS are ordinary anonymous gzip JSONL. FRAME_ROWS is transported '
      'losslessly as binary chunks: concatenate FRAME_SCALAR_MANIFEST.json chunks in list order, '
      'check original bytes/SHA256, then gzip-decode. It has 413427 rows; no scalar is rounded '
      'or omitted. All records contain IoU/reward/delta scalars, not GT geometry or raw predictions.\n\n'
      'Only the missing LN-only arm on the same three actually realized parent permutations '
      'is newly queued (768 fits, 24 full old/new qualification fits). It waits for the existing '
      'baseline, expanded state and deployment timing studies. Queued code/CPU contracts are '
      'not scientific GPU results. The complete reviewer request remains incomplete.\n\n'
      'Private video/RGB, query text, official annotation geometry, model weights, actions, '
      'fit/optimizer payloads and the user manuscript are excluded. The manuscript revision '
      'script, claim-evidence notes and source/output hashes are public; its original project '
      'assets and compiler bundle are unavailable locally.\n')
    entries={}
    def copy(p,rel):
        if p.suffix=='.json':safe(read(p))
        elif p.suffix in {'.py','.md'}:
            t=p.read_text();assert not private_attachment_path(t)
            if p.suffix=='.py':ast.parse(t,filename=rel)
        target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);assert sha(target)==sha(p)
        d=target.read_bytes();blob=hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()
        assert git('hash-object','-w',str(target))==blob
        entries[rel]=dict(path=rel,bytes=len(d),sha256=sha(p),blob_sha=blob)
    for rel in SOURCES:copy(ROOT/rel,rel)
    # Verify that every source dependency pinned in the new runtime is published
    # either by this whitelist or the existing immutable public commit.
    dependency_projections=[]
    for rel,h in pins.items():
        if Path(rel).suffix not in {'.py','.md'}:continue
        p=ROOT/rel;assert sha(p)==h
        dest=CHECKOUT/rel
        if private_attachment_path(p.read_text()):
            # The existing public tree already carries the safe authorization-
            # metadata projection of this historical helper. Keep it unchanged.
            # The scientific source and its original runtime hash stay private.
            assert dest.exists() and not private_attachment_path(dest.read_text())
            assert git('cat-file','-e',base+':'+rel)==''
            ast.parse(dest.read_text(),filename=rel)
            dependency_projections.append(dict(path=rel,original_pinned_source_sha256=h,
                existing_public_source_sha256=sha(dest),
                reason='private authorization attachment paths excluded from existing public projection',
                scientific_source_and_runtime_unchanged=True))
            continue
        if not dest.exists() or sha(dest)!=h:copy(p,rel)
    put(PUB/'PUBLIC_SOURCE_DEPENDENCIES.json',dict(status='public_dependencies_checked',
          private_authorization_metadata_projections_preserved=dependency_projections,
          primary_scientific_source_or_runtime_changed=False))
    for p in sorted(PUB.rglob('*')):
        if p.is_file() and p.name!='FRAME_ROWS.jsonl.gz':copy(p,str(p.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix='stvg-mechanism-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries.values():git('update-index','--add','--cacheinfo','100644',e['blob_sha'],e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
          scope='complete bounded CPU mechanism diagnosis and queued missing LN-only control; not full reviewer request completion',
          base_commit=base,expected_tree=tree,files=[entries[k] for k in sorted(entries)],
          unrelated_untracked_preserved=unrelated,restricted_assets_excluded=True,scalar_record_counts=counts,
          no_new_GPU_result_claim=True,time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',file_count=len(entries),bytes=sum(x['bytes'] for x in entries.values()),
                         chunk_count=len(records),base_commit=base,expected_tree=tree)))


if __name__=='__main__':run()
