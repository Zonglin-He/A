"""Reviewable finite CPU export and GPU preregistration; no commit by this script."""
import gzip,json,shutil,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1';PUB=ROOT/'results/stvg_opd_optimization_geometry/2026-10-12'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'

def project(v):
    if isinstance(v,dict):return {k:project(x) for k,x in v.items() if k not in ['records','prefix','accepted_stage_snapshot','controller','worker','protected','pins','files','complete_witnesses','complete_GPU_fit_witnesses']}
    if isinstance(v,list):return [project(x) for x in v]
    if isinstance(v,str):return v.replace(str(ROOT),'<PROJECT_ROOT>')
    return v

def run():
    for f in ['ROOT_CPU_DIAGNOSTIC_READBACK.json','ROOT_CPU_VISUAL_REVIEW.json']:assert read(BASE/f)['status']=='pass'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=CHECKOUT,text=True).strip()=='https://github.com/Zonglin-He/A.git'
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=CHECKOUT,text=True).strip()
    sources=[p for folder in ['scripts','vg_tta','protocols'] for p in (ROOT/folder).glob('*stvg_opd*geometry*.py')]
    sources += list((ROOT/'protocols').glob('stvg_opd_optimization_geometry*.md'))
    sources += [ROOT/'scripts/pause_stvg_opd_for_geometry_v1.py']
    projections={};owned=set()
    for p in sorted(set(sources)):
        rel=str(p.relative_to(ROOT));content=p.read_text()
        if private_attachment_path(content):
            import re
            content=re.sub(r'/home/[^/]+/\.codex/attachments/[^\"\']+','<LOCAL_AUTHORIZATION_ATTACHMENT>',content)
        assert not private_attachment_path(content)
        target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
        projections[rel]=dict(private_source_sha256=sha(p),public_source_sha256=sha(target),authorization_path_only_projected=sha(p)!=sha(target))
        owned.add(rel)
    receipts=['DESIGN_LOCK.json','CPU_CONTRACTS.json','CPU_RUNTIME.json','CPU_LAUNCH.json','CPU_COMPLETION.json',
      'ROOT_CPU_DIAGNOSTIC_READBACK.json','CPU_ASSOCIATION_READBACK.json','CPU_FIGURE_EXPORT.json','ROOT_CPU_VISUAL_REVIEW.json',
      'GPU_RUNTIME.json','GPU_LAUNCH.json','PRIORITY_AUTHORIZATION.json','BASELINE_PAUSE_RECEIPT.json','HOURLY_MONITOR_SCOPE_UPDATE.json',
      'recovery/preCPU_fixture001/CAPTURE_RECEIPT.json','CPU_readback_schema002/CAPTURE_RECEIPT.json','CPU_readback_schema002/REVISION_RUNTIME.json',
      'CPU_readback_schema002/LAUNCH.json','GPU_hook_binding002/CAPTURE_RECEIPT.json','GPU_hook_binding002/REVISION_RUNTIME.json',
      'GPU_recorder_dispatch003/CAPTURE_RECEIPT.json','GPU_recorder_dispatch003/REVISION_RUNTIME.json',
      'GPU_state_arithmetic004/CAPTURE_RECEIPT.json','GPU_state_arithmetic004/CPU_ACTUAL_WITNESS_READBACK.json','GPU_state_arithmetic004/REVISION_RUNTIME.json',
      'GPU_objective_dispatch005/CAPTURE_RECEIPT.json','GPU_objective_dispatch005/CPU_ACTUAL_DIRECT_READBACK.json',
      'GPU_objective_dispatch005/REVISION_RUNTIME.json','GPU_objective_dispatch005/LAUNCH.json','postseal_CPU_continuation001/RUNTIME.json','postseal_CPU_continuation001/LAUNCH.json']
    for f in receipts:
        p=BASE/f;v=dict(original_private_receipt_sha256=sha(p),projection=project(read(p)));safe(v)
        out=PUB/'receipts'/f;write(out,v)
    write(PUB/'SOURCE_PROJECTION_MANIFEST.json',dict(status='pass',sources=projections,scope='exact science code; local human-authorization attachment path redacted only'))
    count={}
    for f in ['ROUND_GEOMETRY.jsonl.gz','CELL_GEOMETRY.jsonl.gz','CELL_GEOMETRY_WITH_EXISTING_GT_DIAGNOSIS.jsonl.gz']:
        n=0
        with gzip.open(PUB/f,'rt') as stream:
            for s in stream:safe(json.loads(s));n+=1
        count[f]=n
    assert count=={'ROUND_GEOMETRY.jsonl.gz':149610,'CELL_GEOMETRY.jsonl.gz':1792,'CELL_GEOMETRY_WITH_EXISTING_GT_DIAGNOSIS.jsonl.gz':1792}
    source=PUB/'ROUND_GEOMETRY.jsonl.gz';raw=source.read_bytes();chunks=[]
    for i,at in enumerate(range(0,len(raw),512*1024)):
        f=PUB/'round_scalar_chunks'/f'ROUND_GEOMETRY.jsonl.gz.part{i:03}';f.parent.mkdir(exist_ok=True);f.write_bytes(raw[at:at+512*1024])
        chunks.append(dict(path=str(f.relative_to(PUB)),bytes=f.stat().st_size,sha256=sha(f)))
    assert b''.join((PUB/c['path']).read_bytes() for c in chunks)==raw
    write(PUB/'ROUND_SCALAR_MANIFEST.json',dict(status='lossless_original_anonymous_scalar_gzip',rows=149610,original_bytes=len(raw),original_sha256=sha(source),chunks=chunks,
      concatenate_chunks_in_list_order_then_gzip_decode=True,no_scalar_rounding_or_omission=True))
    for ext in ['pdf','png']:shutil.copy2(ROOT/'output/pdf/stvg_opd_optimization_geometry'/('output_geometry.'+ext),PUB/('output_geometry.'+ext))
    write(PUB/'FINITE_CPU_SCOPE.json',dict(status='CPU_diagnostic_root_view_complete_common_state_GPU_pending',CPU_cells=1792,old_complete_fit_byte_reads=3584,
      original_rounds=44800,round_observation_rows=149610,independent_CPU_scalar_checks=907648,new_CPU_model_calls=0,
      GPU_qualification_fits_planned=168,common_state_actual_GPU_fits_planned=5376,new_Direct_formal_predictions_planned=3584,
      common_state_GPU_complete=False,new_common_state_GT_scoring_complete=False,whole_request_complete=False,
      original_baseline_user_paused_predictions=3755,main_method_changed=False))
    (PUB/'README.md').write_text('# OPD optimization geometry\n\nThis batch closes only the finite old-cache CPU diagnostic and publishes the new common-state GPU protocol/code while that experiment is running. Read REPORT.md and FINITE_CPU_SCOPE.json. ROUND_SCALAR_MANIFEST.json describes lossless 512KiB gzip segments; concatenate in the listed order. All scalar records, undefined cosines and negative outcomes are retained. Source projections redact only the local authorization attachment pathname, with both SHA256 values recorded. Restricted raw fits, optimizer tensors, candidate/GT boxes, captions, media and weights remain private. The GPU code requires the original local locked artifacts; it is not a portable model-free rerun. No future GPU result is filled into this batch.\n')
    for p in PUB.rglob('*'):
        if not p.is_file() or p.name=='ROUND_GEOMETRY.jsonl.gz':continue
        rel=str(p.relative_to(ROOT));dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);owned.add(rel)
    items=[]
    for rel in sorted(owned):
        p=CHECKOUT/rel
        gitblob=subprocess.check_output(['git','hash-object',str(p)],cwd=CHECKOUT,text=True).strip()
        items.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),git_blob_sha=gitblob))
    write(BASE/'CPU_PUBLIC_STAGE_MANIFEST.json',dict(status='reviewed_ready_for_exact_remote_commit',scope='finite CPU diagnostic plus GPU preregistration, not new GPU scores',
      files=items,file_count=len(items),bytes=sum(r['bytes'] for r in items),GT_payload_exported=False,time=time.time()))
    print('GEOMETRY_CPU_PUBLIC_READY',len(items),sum(r['bytes'] for r in items),flush=True)

if __name__=='__main__':run()
