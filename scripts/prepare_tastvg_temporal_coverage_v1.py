"""Bind cached A/heads/original critic evidence without reading labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_coverage_common_v1 import *

def run():
    import torch, collections
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(PREVIOUS/'FINAL_COMPLETION.json')['status']=='completed_and_verified_publication'
    sys.addaudithook(guard); tick=time.monotonic()
    cells=read(OLD/'COHORT.json')['cells']
    assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    inputs={};metadata={};counts=collections.Counter()
    def bind(f):
        name=str(f.relative_to(ROOT))
        if name not in inputs: inputs[name]=sha(f)
    for f in [ROOT/'methods/CURRENT_METHOD.json',OLD/'COHORT.json',PREVIOUS/'FINAL_COMPLETION.json']:
        metadata[str(f.relative_to(ROOT))]=sha(f)
    for ds in DATASETS:
        p=plan(ds); assert len(p['rows'])==48
        metadata[str((OLD/ds/'PLAN.json').relative_to(ROOT))]=sha(OLD/ds/'PLAN.json')
        metadata[str((POOL/ds/'CAPTURE_BARRIER.json').relative_to(ROOT))]=sha(POOL/ds/'CAPTURE_BARRIER.json')
    for done,c in enumerate(cells,1):
        budget();f=oldfile(c);bind(f);bind(f.with_suffix('.json'))
        assert inputs[str(f.relative_to(ROOT))]==read(f.with_suffix('.json'))['sha256']
        f=donorfile(c);bind(f);bind(f.with_suffix('.json'))
        assert inputs[str(f.relative_to(ROOT))]==read(f.with_suffix('.json'))['sha256']
        counts[c['dataset']+'_'+c['split']+'_arrivals']+=1
        if c['scheduled']:
            rfile=POOL/c['dataset']/'capture'/c['condition']/f'{c["parent"]:05}.json'
            r=read(rfile);bind(rfile);cf=POOL/c['dataset']/r['cache'];bind(cf)
            assert inputs[str(cf.relative_to(ROOT))]==r['sha256'] and r['pixel_sha256']==c['pixel_sha256']
            rfile=POOL/c['dataset']/'experts/temporal'/c['condition']/f'{c["parent"]:05}.json'
            r=read(rfile);bind(rfile);ef=POOL/c['dataset']/'experts'/r['cache'];bind(ef)
            assert inputs[str(ef.relative_to(ROOT))]==r['cache_sha256'] and r['pixel_sha256']==c['pixel_sha256']
            counts[c['dataset']+'_'+c['split']+'_expert']+=1
        if done%192==0: print('PREPARE_BIND',done,1152,flush=True)
    write(BASE/'COHORT.json',dict(cells=cells,historical_exposure=True,counts=dict(counts),time=time.time()))
    design=dict(allocation='native+early/middle/late_cross_short/medium+global_long',
        position_divisions=[1/3,2/3],duration_divisions=[1/3,2/3],budget=8,
        source_endpoint_prior='offsetwise_softmax_equal_offset_mass_merged_grid',
        critic='original_max_confidence_times_interval_IoU',native_ties='numpy_argmax_native_first',
        all_arrivals=1152,expert_arrivals=288,params={ds:plan(ds)['params'] for ds in DATASETS},
        max_new_expert_calls=0,max_new_model_calls=0,new_temporal_views=0,GT_read=False,
        source_disjoint_within_current_batch=True,all_historically_exposed=True,
        decision='one_predefined_allocation_no_selection_with_GT',time=time.time())
    write(BASE/'DESIGN.json',design)
    for f in [BASE/'COHORT.json',BASE/'DESIGN.json']:metadata[str(f.relative_to(ROOT))]=sha(f)
    previous=read(PREVIOUS/'RUNTIME_LOCK.json')
    labels={f:h for f,h in previous['protected_metadata'].items() if 'GT_LABELS' in f}
    code=['protocols/tastvg_temporal_candidate_coverage_v1.md',
        'docs/tastvg_temporal_candidate_coverage_v1/EXECUTION.md',
        'vg_tta/tastvg_temporal_candidate_coverage_v1.py',
        'scripts/tastvg_temporal_coverage_common_v1.py','scripts/prepare_tastvg_temporal_coverage_v1.py',
        'scripts/generate_tastvg_temporal_coverage_v1.py','scripts/score_tastvg_temporal_coverage_v1.py',
        'scripts/test_tastvg_temporal_coverage_v1.py','vg_tta/tastvg_temporal_qualification_v1.py',
        'vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_paper48_metrics_v1.py',
        'vg_tta/tastvg_paper48_hc2_metrics_v1.py','scripts/tastvg_oracle_event5_common_v1.py',
        'scripts/tastvg_correction_views_common_v1.py','scripts/decota_matrix_common_v1.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in code},inputs=inputs,
        protected_metadata=metadata,label_hashes_from_predecessor_receipt=labels,
        checkpoint_state_sha256={ds:read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'] for ds in DATASETS},
        private_input_count=len(inputs),GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_label_free_generation',GT_read=False,
        counts=dict(counts),time=time.time()))
    assert not torch.cuda.is_initialized()
    write(BASE/'PREPARATION.json',dict(status='pass',GT_read=False,CUDA_initialized=False,
        protected_input_files=len(inputs),worker_wall_seconds=time.monotonic()-tick,time=time.time()))
    # Archive subprocess reads its own ledger; the current process remains label-free.
    archive('原始名单、A/heads/critic缓存及唯一分配规则已预锁，尚未生成新池或读取本轮GT')
    print('PREPARED',dict(counts),flush=True)
if __name__=='__main__':run()
