"""Explicit anonymous public export manifest, with private input hashes only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_coverage_common_v1 import *

def run():
    verify(inputs=True,labels=True)
    assert read(BASE/'FINAL_ROOT_AUDIT.json')['status']=='pass'
    runtime=read(BASE/'RUNTIME_LOCK.json');cohort=read(BASE/'COHORT.json')
    write(PUBLIC/'CONFIG.json',dict(version='tastvg_temporal_candidate_coverage_v1',
        predecessor_commit='9485d3673a7a1320e03026b75b6075856322e4c0',
        design=read(BASE/'DESIGN.json'),checkpoint_state_sha256=runtime['checkpoint_state_sha256'],
        production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        sampling='original_Paper48_observed_grid',confirmation='historical_source_disjoint_within_current_batch',
        source_bootstrap_draws=10000,source_bootstrap_seed=20261003,
        anonymous_source_ordinals={ds:{split:sorted({c['parent'] for c in cohort['cells'] if c['dataset']==ds and c['split']==split})
            for split in SPLITS} for ds in DATASETS},
        no_extra_GPU_or_expert_calls=True,no_new_loss_or_view=True,production_unchanged=True))
    write(PUBLIC/'RESOURCES.json',dict(preparation=read(BASE/'PREPARATION.json'),
        generation=read(BASE/'GENERATION_RESOURCES.json'),score=read(BASE/'SCORE_ROOT_CHECKS.json'),
        root=read(BASE/'FINAL_ROOT_AUDIT.json')))
    extra=['scripts/audit_tastvg_temporal_coverage_root_v1.py','scripts/audit_tastvg_temporal_coverage_public_v1.py',
        'scripts/report_tastvg_temporal_coverage_v1.py','scripts/export_tastvg_temporal_coverage_v1.py']
    pins={f:sha(ROOT/f) for f in sorted(set(list(runtime['pins'])+extra))}
    write(PUBLIC/'RUNTIME_PROVENANCE.json',dict(pins=pins,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        private_immutable_input_count=len(runtime['inputs']),private_assets_not_exported=True,
        labels_used_for_scoring_only_after_global_prediction_seal=True))
    b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    write(PUBLIC/'BARRIERS.json',dict(global_prediction={k:v for k,v in b.items() if k!='files'},
        GT_exposure=read(BASE/'GT_EXPOSURE.json'),root=read(BASE/'FINAL_ROOT_AUDIT.json')))
    write(PUBLIC/'ENGINEERING_HISTORY.json',dict(science_changed=False,recoveries=[
        dict(name=f.name,record=read(f/'FAILURE.json'),original_sources_preserved=True)
        for f in sorted((BASE/'recovery').glob('*'))]))
    files=list(pins)+['docs/TA_TEMPORAL_CANDIDATE_COVERAGE_REVIEW.md']
    files += [str(f.relative_to(ROOT)) for f in PUBLIC.rglob('*') if f.is_file()]
    files=sorted(set(files));assert all(not any(s in f for s in ['GT_LABELS','.pt','/expert_cache/','/checkpoints/']) for f in files)
    def scan(value):
        if isinstance(value,dict):
            for k,v in value.items():
                assert k.lower() not in {'caption','query_text','truth','gt_span','gt_interval','weights','gradients','h','pre_state','post_state'},k
                scan(v)
        elif isinstance(value,list):
            for v in value:scan(v)
    for f in PUBLIC.rglob('*.json'):scan(read(f))
    write(BASE/'PUBLIC_MANIFEST.json',dict(files={f:dict(sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files},
        file_count=len(files),bytes=sum((ROOT/f).stat().st_size for f in files),time=time.time()))
    print('PUBLIC_ALLOWLIST',len(files),'files',flush=True)
if __name__=='__main__':run()
