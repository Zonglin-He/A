"""Reviewed finite whitelist stage, prior to authenticated GitHub writes."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_opd_supplement_v1';PUB=ROOT/'results/stvg_opd_supplement/2026-10-11'
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-paper-hc2-revision-v2'

def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()

def private_attachment_path(value):
    return re.search(r'/(?:home|Users)/[^/\s]+/\.codex/attachments/',value) is not None

def safe(value):
    forbidden={'caption','query_text','video_path','native_boxes','GT_boxes','GT_box','GT_geometry','committed','gradient','optimizer_state','raw_logits','hidden_states','actions','actual_actions','actual_samples','sample_logits','sample_boxes','frame_pixels','weights','Fisher','closure_evidence'}
    if isinstance(value,dict):
        assert not set(value)&forbidden,set(value)&forbidden
        for v in value.values():safe(v)
    elif isinstance(value,list):
        for v in value:safe(v)
    elif isinstance(value,str):assert not private_attachment_path(value)

def run():
    assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    for ns in ['orders','baselines','cost']:assert (BASE/ns/'ROOT_FIGURE_COMPLETION.json').exists()
    assert read(BASE/'orders/ROOT_ALL_FITS_AND_STATISTICS.json')['complete_fit_math']==2304
    assert read(BASE/'baselines/ROOT_ALL_SELECTED_FITS_READBACK.json')['selected_logical_arrivals']==82710
    assert read(PUB/'cost/END_TO_END_SUMMARY.json')['query_count']==32
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();assert all('stvg_motivation' in p.lower() for p in unrelated)
    owned=set();bindings={};projections={}
    def copy(p,rel):
        assert p.suffix in {'.json','.jsonl','.gz','.md','.png','.pdf','.py'}
        if p.suffix=='.json':safe(read(p))
        if p.suffix=='.gz':
            import gzip
            with gzip.open(p,'rt') as f:
                for line in f:safe(json.loads(line))
        target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(target)==sha(p);owned.add(rel)
        bindings[rel]=dict(source_path=str(p.relative_to(ROOT)),source_sha256=sha(p),public_sha256=sha(target),bytes=target.stat().st_size)
    for p in sorted(PUB.rglob('*')):
        if p.is_file() and 'receipts' not in p.parts:copy(p,str(p.relative_to(ROOT)))
    sources=['protocols/stvg_opd_supplement_v1.md','protocols/stvg_opd_end_to_end_cost_supplement_v1.md']
    sources += ['scripts/stvg_opd_order_supplement_common_v1.py','scripts/run_stvg_opd_order_supplement_v1.py','scripts/continue_stvg_opd_order_supplement_v1.py','scripts/score_stvg_opd_order_supplement_v1.py','scripts/finalize_stvg_opd_order_supplement_v1.py','scripts/stvg_opd_baseline_supplement_common_v1.py','scripts/run_stvg_opd_baseline_supplement_v1.py','scripts/continue_stvg_opd_baseline_supplement_v1.py','scripts/score_stvg_opd_baseline_supplement_v1.py','scripts/analyze_stvg_opd_supplement_saved_v1.py','scripts/render_stvg_opd_supplement_cases_v1.py','scripts/run_stvg_opd_end_to_end_cost_supplement_v1.py','scripts/continue_stvg_opd_end_to_end_cost_supplement_v1.py','scripts/review_stvg_opd_supplement_v1.py','scripts/continue_stvg_opd_supplement_root_v1.py','scripts/audit_stvg_opd_supplement_public_v1.py','scripts/publish_stvg_opd_supplement_v1.py','scripts/verify_stvg_opd_public_remote_v1.py']
    for rel in sorted(set(sources)):
        p=ROOT/rel;text=p.read_text();assert not private_attachment_path(text)
        if p.suffix=='.py':ast.parse(text)
        copy(p,rel)
    receipts=['ROOT_VISUAL_REVIEW.json','ROOT_REVIEW.json','saved_evidence/ROOT_VISUAL_REVIEW.json','saved_evidence/COMPLETION.json','saved_evidence/CASE_COMPLETION.json','actual_root/RUNTIME.json']
    for ns in ['orders','baselines','cost']:
        receipts += [f'{ns}/{p.name}' for p in (BASE/ns).glob('*.json') if p.name not in {'STATUS.json','PROGRESS.json'}]
        receipts += [str(p.relative_to(BASE)) for p in (BASE/ns).rglob('REVISION_RUNTIME.json')]
        receipts += [str(p.relative_to(BASE)) for p in (BASE/ns).rglob('CAPTURE_RECEIPT.json')]
        receipts += [str(p.relative_to(BASE)) for p in (BASE/ns).rglob('CPU_CONTRACTS.json')]
    for basename in ['REVISION_RUNTIME.json','CAPTURE_RECEIPT.json','CPU_CONTRACTS.json']:
        receipts += [str(p.relative_to(BASE)) for p in (BASE/'actual_root').rglob(basename) if 'original' not in p.parts]
    for rel in sorted(set(receipts)):
        p=BASE/rel;v=read(p)
        # Preserve integrity metadata but withhold private native-head VJP/error records.
        def project(x):
            if isinstance(x,dict):return {k:project(v) for k,v in x.items() if k not in ['records','native_head_VJP','math_audit','full_payloads','private_PDF_path']}
            if isinstance(x,list):return [project(v) for v in x]
            if isinstance(x,str):return x.replace(str(ROOT),'<PROJECT_ROOT>')
            return x
        projected=project(v)
        if rel=='baselines/DESIGN_LOCK.json':
            for ds,definition in projected['datasets'].items():
                plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
                ids={name:i for i,name in enumerate(sorted({r['source'] for r in plan['rows']}))}
                raw=definition.pop('excluded_primary_parent_sources')
                definition['excluded_primary_anonymous_source_ids']=sorted(ids[name] for name in raw)
                definition['excluded_private_parent_roster_sha256']=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
                definition.pop('development_parents',None)
        target=PUB/'receipts'/rel;write(target,dict(original_receipt_sha256=sha(p),projection=projected));safe(read(target));copy(target,str(target.relative_to(ROOT)));projections[rel]=sha(p)
    metadata=PUB/'PUBLIC_INPUT_AND_SOURCE_BINDINGS.json';write(metadata,dict(status='reviewed_actual_results_stage',bindings=bindings,private_receipt_bindings=projections,restricted_assets_excluded=True,private_RGB_case_PDF_not_public=True,GT_geometry_and_native_tensors_not_public=True));copy(metadata,str(metadata.relative_to(ROOT)))
    result=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/audit_stvg_opd_supplement_public_v1.py'],cwd=CHECKOUT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr;audit=json.loads(result.stdout);assert audit['status']=='pass';write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()))
    entries=[]
    for rel in sorted(owned):
        p=CHECKOUT/rel;data=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();entries.append(dict(path=rel,sha256=sha(p),bytes=len(data),blob_sha=blob))
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='all actual additional orders, calibrated baselines, end-to-end cost and saved evidence figures',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,files=entries,unrelated_untracked_preserved=unrelated,public_scalar_audit_sha256=sha(BASE/'PUBLIC_SCALAR_AUDIT.json'),time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',files=len(entries),bytes=sum(x['bytes'] for x in entries),base_commit=base)))

if __name__=='__main__':run()
