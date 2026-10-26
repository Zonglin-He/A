"""Two retained HC CG grids: conservative control exact and full-spatial replay.

This is a no-model/no-GT numerical preflight, not a full-domain score or GPU
bridge. The active CG owner, original profiles and all frozen files are intact.
"""
from review_common import *
import argparse

PROFILE=dict(calibration='Transferred',gamma=.5,unobserved=.5,link=.25,
             beta=.5,kappa=0,context=1.5)
PLAN=D/'CG_full_spatial_preflight_plan_v1.json'


def load_modules():
    path=R/'studies/shared_corrected_roi_CG_v1'
    sys.path.insert(0,str(path))
    import cg_common as cg
    import cg_core
    cg.validate()
    return cg,cg_core


def prepare():
    assert not PLAN.exists();cg,core=load_modules()
    files=[Path(__file__),S/'review_common.py',R/'studies/shared_corrected_roi_CG_v1/cg_common.py',
           R/'studies/shared_corrected_roi_CG_v1/cg_core.py',R/'studies/cgstvg_full_v1/core.py',
           R/'studies/paper_review_suite_v1/cgstvg_interfaces_v9.py',
           R/'stvg/initialization.py',R/'stvg/predictor.py',R/'stvg/spatial.py',
           R/'stvg/evidence.py',R/'stvg/prototype.py']
    inputs=[D/'intent.json',cg.D/'plan.json',cg.D/'hc2_sealed.json',
            cg.D/'hc2_reporting_receipt.json',cg.CGOLD/'reuse_index.json',
            R/'outputs/paper_followup_v1/cg_calibrations.json']
    p=dict(schema='predeclared_CG_HC_complete_spatial_CPU_preflight_v1',
           profile=PROFILE,scope='Two original retained HC raw grids and original eight expert contexts only; no new Source/model/expert/GPU/optimizer/GT.',
           controls='Recover current conservative HC complete arrays/old native logits/q0/old profile pool exactly first; independently rebuild gamma.5 corrected boxes, bidirectional geometry, actual native E-ROI pool, kappa0/beta.5 posterior, jointMAP and final DP with observed/unobserved .5/.5 and link.25. No first auxiliary DP in selected full-spatial path. Both repeats all arrays exact; Torch4 cold replay tested separately.',
           no_tuning='Only the already predeclared gamma.5/u.5/link.25 enablement; HC beta.5/context1.5 unchanged. Not selected on full outcomes. CG-Vid current profile already full-spatial and must not be rerun.',
           future='Full HC1902 requires separate acquisition/execution freeze, inventory of missing raw and necessary at-most-once CG420 Source with checkpoint-before-teacher/ROI and original native/old-pool exact gates. No new queue or GPU owner is created here.',
           code_sha256={str(f.relative_to(R)):sha(f) for f in files},
           input_sha256={str(f.relative_to(R)):sha(f) for f in inputs})
    p['fingerprint']=digest(p);atomic(PLAN,p);atomic(P/'CG_full_spatial_preflight_protocol_v1.json',p)


def derive(cap,teacher,profile):
    import numpy as np
    from stvg.initialization import initialize
    from stvg.predictor import output_writeback
    from stvg.evidence import SourceEvidence
    from stvg.spatial import SpatialGeometry
    from stvg.prototype import one_step
    from stvg.current import map_interval
    from studies.cgstvg_full_v1.core import solve
    from cgstvg_interfaces_v9 import pool_single_grid
    from cgstvg_runtime_v9 import clip_native_boxes
    ref=np.asarray(teacher['boxes'],np.float64);ids=cap['frame_ids'];pos=cap['positions8']
    calibration=read(R/'outputs/paper_followup_v1/cg_calibrations.json')['hc2'][profile['calibration']]
    raw,initialization=initialize(cap['Source'],pos,ref,calibration)
    E=output_writeback(cap['Source'],raw,profile['gamma']);grid=cap['grids']
    evidence=SourceEvidence.from_native_masks(grid['spatial'],grid['spatial_valid'],ids,pos,
        teacher['native_mask_logits'],teacher['input_shape'][-2:],grid['shape'],selected_query=teacher['selected_query'])
    geometry=SpatialGeometry(evidence,E,ref)
    poolboxes=E.copy();poolboxes[:,2:]*=profile['context']/1.5
    features=pool_single_grid(grid['projected'],grid['valid'],poolboxes)
    q,_=one_step(features,cap['q0'],kappa=0,beta=profile['beta'])
    iv=map_interval(ids,q);boxes=clip_native_boxes(solve(geometry,iv,profile))
    bank=geometry.for_interval(iv)
    return dict(Source=np.asarray(cap['Source']),E=E,features=features,posterior=q,
                interval=np.asarray(iv),boxes=boxes,candidates=bank.boxes.numpy(),
                candidate_valid=bank.valid.numpy(),unary=bank.score_unary.numpy(),links=bank.score_pair.numpy())


def run():
    import numpy as np,torch
    gt_guard();torch.set_num_threads(1);p=read(PLAN)
    assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    cg,core=load_modules();old=cg.legacy();reuse=read(cg.CGOLD/'reuse_index.json')['hc2']
    rows={x['id']:x for x in old.rows('hc2')};proofs=[]
    for key,e in list(reuse.items())[:2]:
        cap,native,origin=old.load_legacy(e,rows[key],'hc2')
        before,compact,proof,cost=core.build(cap,native,'hc2',key)
        ce=read(cg.D/'hc2_sealed.json')['files'][key]
        assert sha(cg.D/ce['path'])==ce['sha256'] and sha(cg.D/ce['receipt_path'])==ce['receipt_sha256']
        with np.load(cg.D/ce['path']) as z:assert all(np.array_equal(before[k],z[k]) for k in before)
        a=derive(cap,native,PROFILE);b=derive(cap,native,PROFILE)
        assert all(np.array_equal(a[k],b[k]) for k in a)
        torch.set_num_threads(4)
        try:c=derive(cap,native,PROFILE)
        finally:torch.set_num_threads(1)
        assert all(np.array_equal(a[k],c[k]) for k in a)
        assert a['unary'].shape[0]==len(cap['frame_ids'])
        proofs.append(dict(original_raw_capture_sha256=origin['capture_sha256'],
                           current_conservative_prediction_sha256=ce['sha256'],
                           original_teacher_sha256=origin['teacher_sha256'],
                           conservative_control_all_arrays_exact=True,
                           actual_changed_E_native_grid_ROI_pool=True,
                           rebuilt_bidirectional_geometry=True,
                           full_spatial_repeat_all_arrays_exact=True,
                           Torch1_Torch4_all_arrays_exact=True,
                           final_DP_retained=True,first_aux_DP_in_selected_path=False))
    assert not torch.cuda.is_initialized()
    r=dict(status='passed_retained_raw_CPU_only',profile=PROFILE,cases=proofs,
           fingerprint=p['fingerprint'],GT_read=False,new_Source=0,new_model=0,new_expert=0,
           new_optimizer=0,new_GT_scoring=0,actual_new_GPU_bridge=False,
           full_HC_score=False,active_CG_owner_untouched=True,worker=identity())
    atomic(D/'CG_full_spatial_preflight_receipt_v1.json',r)
    atomic(P/'CG_full_spatial_preflight_receipt_v1.json',r)
    print(json.dumps(dict(status=r['status'],cases=len(proofs),new_Source=0)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run'])
    globals()[a.parse_args().command]()
