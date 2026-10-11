"""Whitelist stage for viewed historical cases and predeclared new experiments.

This stage publishes no RGB, query text, GT geometry, native box arrays or fit
payloads. New state/latency experiments remain explicitly incomplete here.
The authenticated connector creates the remote commit after review.
"""
import ast
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
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1'
PUB=ROOT/'results/stvg_opd_claim_strengthening/2026-10-11'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'
SOURCES=['protocols/stvg_opd_claim_strengthening_v1.md',
    'scripts/check_stvg_opd_claim_strengthening_cpu_v1.py',
    'scripts/stvg_opd_expanded_state_common_v1.py','scripts/run_stvg_opd_expanded_state_v1.py',
    'scripts/score_stvg_opd_expanded_state_v1.py','scripts/finalize_stvg_opd_expanded_state_v1.py',
    'scripts/run_stvg_opd_deployment_latency_v1.py','scripts/finalize_stvg_opd_deployment_latency_v1.py',
    'scripts/continue_stvg_opd_claim_strengthening_v1.py','scripts/render_stvg_opd_aligned_observation_cases_v1.py',
    'vg_tta/stvg_opd_deployment_profile_v1.py','scripts/publish_stvg_opd_claim_strengthening_v1.py']


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=CHECKOUT,env=env,text=True).strip()


def projection(x):
    if isinstance(x,dict):
        return {k:projection(v) for k,v in x.items() if k not in
                ['records','frames','native_head_VJP','math_audit','full_payloads',
                 'excluded_development_parent_sources','query_text','caption']}
    if isinstance(x,list):return [projection(v) for v in x]
    if isinstance(x,str):return x.replace(str(ROOT),'<PROJECT_ROOT>')
    return x


def run():
    from scripts.continue_stvg_opd_claim_strengthening_v1 import verify
    verify();assert read(BASE/'aligned_cases/ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD')
    assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in p.lower() for p in unrelated)
    entries={}
    def copy(p,rel):
        assert p.suffix in {'.json','.md','.py'}
        if p.suffix=='.json':safe(read(p))
        else:
            assert not private_attachment_path(p.read_text())
            if p.suffix=='.py':ast.parse(p.read_text(),filename=rel)
        target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);assert sha(target)==sha(p)
        data=target.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        entries[rel]=dict(path=rel,sha256=sha(p),bytes=len(data),blob_sha=blob)
        actual=git('hash-object','-w',str(target));assert actual==blob
    for rel in SOURCES:copy(ROOT/rel,rel)
    private=read(BASE/'aligned_cases/PRIVATE_READBACK.json')
    cases=[dict(anonymous_case_id=i,dataset=r['dataset'],historical_role=r['historical_role'],
                original_fit_sha256=r['original_fit_sha256'],original_input_sha256=r['original_input_sha256'],
                original_pixel_sha256=r['pixel_sha256'],displayed_frame_positions=7,
                nominal_expert_positions=4,intermediate_original_positions=3,panels=r['panels'],
                new_model_calls=0,new_GT_experiment=False)
           for i,r in enumerate(private['records'])]
    write(PUB/'ALIGNED_CASES_PROVENANCE.json',dict(status='actual_CPU_render_and_PDF_pixel_view_pass',
        cases=cases,case_count=3,panels=84,new_model_calls=0,private_media_not_public=True,
        official_GT_geometry_private=True,posthoc_illustrative_not_population_evidence=True,
        private_PDF_sha256=private['pdf_sha256']))
    receipts=['CONTROLLER_RUNTIME.json','LAUNCH.json','CPU_IMPLEMENTATION_READBACK.json',
        'expanded_state/DESIGN_LOCK.json','expanded_state/RUNTIME_LOCK.json','expanded_state/CPU_CONTRACTS.json',
        'deployment_latency/DESIGN_LOCK.json','deployment_latency/RUNTIME_LOCK.json','deployment_latency/CPU_CONTRACTS.json',
        'deployment_latency/preGPU_receipt_dispatch002/REVISION_RUNTIME.json',
        'deployment_latency/preGPU_receipt_dispatch002/CAPTURE_RECEIPT.json',
        'deployment_latency/preGPU_receipt_dispatch002/CPU_IMPLEMENTATION_READBACK.json',
        'deployment_latency/preGPU_receipt_dispatch002/ORIGINAL_SAVED_MATH_READBACK.json',
        'preGPU_CPU_contract_001/CAPTURE_RECEIPT.json',
        'aligned_cases/CASE_LOCK.json','aligned_cases/RUNTIME_LOCK.json','aligned_cases/ROOT_VISUAL_REVIEW.json']
    for rel in receipts:
        p=BASE/rel;value=projection(read(p))
        # Input-binding tensors stay private; metadata retains their collective
        # runtime SHA and the source-only pins that can be inspected publicly.
        if 'pins' in value:value['pins']={k:v for k,v in value['pins'].items() if not k.endswith('.pt')}
        if rel=='expanded_state/DESIGN_LOCK.json':
            for ds,cohort in value['cohorts'].items():
                plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
                ids={s:i for i,s in enumerate(sorted({r['source'] for r in plan['rows']}))}
                raw=read(p)['cohorts'][ds]['excluded_development_parent_sources']
                cohort['excluded_development_anonymous_source_ids']=[ids[s] for s in raw]
        target=PUB/'receipts'/rel
        write(target,dict(original_private_receipt_sha256=sha(p),projection=value))
    write(PUB/'PUBLIC_STAGE_STATUS.json',dict(
        status='qualitative_cases_viewed_new_GPU_experiments_predeclared_and_queued',
        expanded_state_new_formal_fits_planned=5430,nondevelopment_parents=905,
        expanded_state_new_formal_fits_completed=0,expanded_qualification_fits_planned=48,
        deployment_unique_queries_planned=32,deployment_qualification_fits_planned=64,
        deployment_timed_paths_planned=128,deployment_timed_paths_completed=0,
        new_state_GT_scores_read=False,new_latency_GT_read=False,
        all_new_experiments_complete=False,paper_claim_strengthening_complete=False))
    (PUB/'README.md').write_text(
        '# Claim-strengthening supplement\n\n'
        'Three original qualitative cases have been rendered and every PDF page visually checked. '
        'Each uses Native / Before / After / GT, all four expert observation positions and three actual '
        'sampled intermediate frames. RGB, query text and GT geometry remain private. Two historical '
        'improvements and one regression are retained; these examples do not estimate prevalence.\n\n'
        'New predeclared GPU work is queued after the existing calibrated baselines and audited cost '
        'readback: Full / alpha0 / alpha1 on 205 HC and 700 Vid non-development parents, two fixed '
        'orders, 5430 new fits; paired Full / Frozen deployment profiling on 32 fixed queries with '
        'two repeats. No new state comparison or deployment timing result exists in this stage. '
        'All future state predictions must globally seal before GT; complete fit qualification '
        'must pass before minimal-path timing. Full-minus-alpha0 and Full-minus-alpha1 are the '
        'direct endpoints. Historical exposure and negative results are retained.\n')
    for p in sorted(PUB.rglob('*')):
        if p.is_file():copy(p,str(p.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix='stvg-claim-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')}
        git('read-tree',base,env=env)
        for e in entries.values():git('update-index','--add','--cacheinfo','100644',e['blob_sha'],e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'PREPARATION_AND_VISUAL_PUBLIC_STAGE.json',dict(status='reviewed_staged',
        scope='three viewed historical qualitative cases and predeclared queued state/paired-latency experiment code; no new GPU results',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,
        files=[entries[p] for p in sorted(entries)],unrelated_untracked_preserved=unrelated,
        restricted_assets_excluded=True,new_GPU_result_claim=False,time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',file_count=len(entries),
        bytes=sum(e['bytes'] for e in entries.values()),base_commit=base,expected_tree=tree)))


if __name__=='__main__':run()
