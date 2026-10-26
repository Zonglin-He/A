"""Freeze full cohorts, original raw acquisition and all-five-chain sensitivity."""
from gamma_full_common import *
import argparse
def prepare():
    assert not(D/'plan.json').exists();review.validate()
    code=[*sorted(S.glob('*.py')),REVIEW/'review_common.py',REVIEW/'gamma_cached_core_v1.py',
          REVIEW/'gamma_cached_v1.py',R/'studies/map_paper_completion_v1/spatial_core.py',
          R/'tools/full_score.py',R/'studies/paper_followup_v1/common.py',*sorted((R/'stvg').glob('*.py'))]
    inputs=[review.D/'intent.json',review.D/'active_scope_after_corruption_deferral.json',
            RETAINED/'plan.json',RETAINED/'root_both_subset_completion.json',
            selected.PREV/'raw_reuse_index.json',RUN/'plan.json',RUN/'reuse_index.json',
            R/'outputs/shared_corrected_roi_CG_v1/plan.json',R/'outputs/shared_corrected_roi_CG_v1/controller.json',
            R/'outputs/maintenance/corruption_paused_by_user_20261026/owned_exit.json']
    for ds in COUNTS:inputs += [RETAINED/ds/'sealed_retained_subset.json',RUN/ds/'sealed.json',
            selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json']
    p=dict(schema='full_current_gamma_sensitivity_v1',cohorts=COUNTS,values=[0,.25,.5,.75,1],
           gamma_default={ds:default_arm(ds)for ds in COUNTS},
           science='Reuse the frozen gamma_cached_core_v1 build/replay unchanged: each gamma rebuilds shared corrected boxes, both-anchor geometry, actual native pretext E-ROI pooling, kappa0/beta1 one prototype pair, legal jointMAP and final trajectory DP. Same original eight experts; no first auxiliary DP in the selected path. Default arm bitwise current full prediction and interval per query. Source is native offset MAP hull, SourceJointMAP separate.',
           reuse='All192 already committed gamma retained-grid outputs/domain are cold-checked and joined to bound native controls; no new raw pooling for these. Remaining exact legacy full grids first; spatial-only legacy permits one native appearance prefix; otherwise exactly one original Source224 full forward. One acquired grid serves all five gamma points. No replacement pooled vectors as raw grids.',
           bounds=dict(max_new_Source_or_prefix_queries=12205-384,max_new_expert=0,max_new_optimizer=0),
           acquisition='Original224/two offsets/native FP32 final heads, same deterministic seed and native FP16 appearance substage. Source box/logit/frame records exact against original; complete old spatial64 arrays and auxiliary ROI pool exact. Every NEW raw.pt+SHA checkpoint before teacher-context verification or ROI. Exact original RGB/eight teacher context/key/payload SHA mandatory; no teacher forward/fallback. New compact all-array cold replay + prediction readback + receipt commit before deleting only own new raw. First two acquisitions remain raw and pause for root bridge acknowledgment. Old retained grids never deleted, failures/uncommitted paid attempts never retried automatically.',
           dependency='Original CG owner3374443/start252798286 actual successful exit, both whole-domain official/cold-array/means audits and root complete_published receipts. No GPU lease/model during waiting. Corruption is paused by user and never a successor here. Existing CG untouched.',
           evaluation='Whole1902/10303 seal before separate CPU official scorer and separate full cold-array/input/teacher/means auditor. All signed comparisons to Source/SourceJointMAP/default, severe t/v union once/rescues, short<=5s/long, parent pointwise95CI and full curves. All target data exposed; descriptive sensitivity only, no parameter/default selection.',
           storage='Retained192 sample projects about8.6GiB HC and82.9GiB Vid gamma compacts; this is an empirical estimate, not a capacity guarantee. Per-query resources and exact full namespace private160GiB hard guards. Sufficient states retain all five geometries/features for exact gamma replay, not arbitrary future raw repooling.',
           cost='Actual PID/start/model load/Source or prefix/legacy/teacher lookup/compatibility/build/query elapsed/whole worker preserved separately; nested costs never added or represented as deployment. Original Source/expert acquisition remains nonzero.',
           limits=dict(RAM_available_bytes=8*2**30,disk_free_bytes=80*2**30,private_bytes=160*2**30,
                       stage_seconds=259200,CPU_threads=1,Source_numeric_threads=4,CPU=[14,15],nice=19,ionice=3,max_CPU_jobs=2),
           code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
    p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
    for f in code:
        q=D/'source_snapshot'/f.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
    print(p['fingerprint'])
def preflight():
    import torch,numpy as np
    from gamma_cached_core_v1 import build
    review.gt_guard();torch.set_num_threads(4);validate();proof=[]
    for ds in COUNTS:
        rows={r['id']:r for r in records(ds)};retained=read(RETAINED/ds/'sealed_retained_subset.json')
        for key,e in list(read(selected.PREV/'raw_reuse_index.json')[ds].items())[:2]:
            c,parent=selected.suite().load_parent(ds,rows[key]);assert sha(e['path'])==e['sha256']
            raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);current,ce=checked_current(ds,key)
            a,compact,default=build(c,raw,ds,current);b,_,_=build(c,raw,ds,current)
            assert all(np.array_equal(a[k],b[k])for k in a)
            torch.set_num_threads(1)
            try:cold=complete_arrays(compact,current)
            finally:torch.set_num_threads(4)
            assert all(np.array_equal(a[k],cold[k])for k in a)
            re=retained['files'][key]
            for field,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(RETAINED/re[field])==re[h]
            with np.load(RETAINED/re['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
            proof.append(dict(dataset=ds,raw_sha256=e['sha256'],current_prediction_sha256=ce['sha256'],all_five_repool_repeat_exact=True,retained_precompute_arrays_exact=True,Torch4_Torch1_cold_exact=True,default_arm=default))
    assert not torch.cuda.is_initialized();rec=dict(status='passed_four_actual_cached_inputs',cases=proof,new_model=0,new_Source=0,new_expert=0,GT_read=False,actual_new_GPU_bridge=False,plan_sha256=sha(D/'plan.json'))
    atomic(D/'preflight_receipt.json',rec);atomic(P/'preflight_receipt.json',rec);print(rec['status'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight']);globals()[a.parse_args().command]()
