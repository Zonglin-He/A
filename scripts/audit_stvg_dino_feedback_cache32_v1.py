"""Independent cached census/geometry/gate audit, without labels or inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,load,put,sha
from scripts.diagnose_stvg_dino_feedback_cache32_v1 import D,P,PRE,verify,read_guard,export,ARM_MAP


def main():
    import torch
    p=verify();read_guard();raw=load(D/'CACHED_SCALAR_ROWS.pt');result=read(P/'DIAGNOSIS.json')
    got_candidates={(x['alias'],x['arm'],x['round'],x['sample']):x for x in raw['candidates']}
    got_anchors={(x['alias'],x['arm'],x['round'],x['sample'],x['position']):x for x in raw['anchors']}
    checks=collections.Counter();cnt=collections.defaultdict(collections.Counter)
    def overlaps(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float).reshape(-1,4)
        if not len(b):return np.zeros(0)
        a0=a[:2]-a[2:]/2;a1=a[:2]+a[2:]/2;b0=b[:,:2]-b[:,2:]/2;b1=b[:,:2]+b[:,2:]/2
        i=np.maximum(np.minimum(a1,b1)-np.maximum(a0,b0),0).prod(1);u=a[2:].prod()+b[:,2:].prod(1)-i
        return np.divide(i,u,out=np.zeros_like(i),where=u>0)
    for r in p['records']:
        z=load(PRE/'predictions'/(r['alias']+'.pt'));src=load(ROOT/r['prior_Source']);e=load(ROOT/r['prior_prediction']);b=np.asarray(e['outputs']['Source_no_correction'],float)
        scales=read(PRE/'PLAN_LOCK.json')['scale'][r['dataset']];lat=np.concatenate([b[:,:2],np.log(b[:,2:])],1)
        for arm,name in ARM_MAP.items():
            for st in z['fits'][name]['rounds']:
                details=st['details'][:1]if arm=='One_at_E'else st['details']
                for k,detail in enumerate(details):
                    key=(r['alias'],arm,st['round'],k);row=got_candidates[key];x=lat+np.array(detail['action'])*scales
                    refs=np.concatenate([x[:,:2],np.exp(x[:,2:])],1)[z['planned_positions']]
                    legal=np.isfinite(refs).all(1)&(refs>0).all(1)&(refs<1).all(1)
                    assert row['legal']==bool(legal.all())==detail['legal']
                    assert row['legal_anchor_count']==int(legal.sum())and row['invalid_anchor_count']==int((~legal).sum())
                    assert row['individually_valid_anchors_discarded']==(int(legal.sum())if not legal.all()else 0)
                    for j,n in enumerate(['invalid_cx','invalid_cy','invalid_width','invalid_height']):assert row[n]==int((~((refs[:,j]>0)&(refs[:,j]<1))).sum())
                    checks['candidate_legality_and_coordinate_counts']+=1
                    tally=cnt[(r['dataset'],arm)];tally['candidates']+=1;tally['legal']+=bool(legal.all());tally['invalid']+=not legal.all();tally['feedback']+=detail['accepted_anchors']>0
                    if not legal.all():continue
                    for j,pos in enumerate(z['planned_positions']):
                        obs=src['expert']['observations'][('original',pos)];probe=obs['probe'];d=detail['anchors'][pos];a=got_anchors[key+(pos,)];ref=refs[j]
                        assert a['score']==d['score']and a['slot']==d['slot']and a['accepted']==(d['score']>=.35)
                        assert a['input_reference_fp32_digest']==hashlib.sha256(ref.astype(np.float32).tobytes()).hexdigest()
                        b0=ref[:2]-ref[2:]/2;b1=ref[:2]+ref[2:]/2;visible=float(np.maximum(np.minimum(b1,1)-np.maximum(b0,0),0).prod()/ref[2:].prod())
                        assert abs(visible-a['student_visible_fraction'])<1e-12
                        pp=np.asarray(probe['boxes'],float).reshape(-1,4);native=overlaps(ref,pp)
                        if len(pp):assert abs(native.max()-a['input_native_max_iou'])<1e-12
                        matched=list(probe['candidate_ids']).index(d['slot'])if obs['receipt']['context_active']and d['slot']in probe['candidate_ids']else None
                        expected=None if matched is None else float(probe['target_scores'][matched]);assert expected==a['matched_native_slot_score']
                        tally['anchor_refs']+=1;tally['anchor_accept']+=d['score']>=.35;tally['near035']+=.30<=d['score']<.35
                        checks['anchor_scores_slots_geometry']+=1
    for (ds,arm),t in cnt.items():
        g=result['groups'][ds][arm]
        assert t['candidates']==g['candidate_trajectories']and t['legal']==g['legal_trajectories']and t['invalid']==g['illegal_trajectories']
        assert t['anchor_refs']==g['anchor_scores']['n']and t['anchor_accept']==g['anchor_scores']['accepted']and t['near035']==g['anchor_scores']['near_below_030_035']
        assert t['feedback']==g['legal_trajectories_with_feedback']
        vals=[x['score']for x in raw['anchors']if x['dataset']==ds and x['arm']==arm]
        q=np.quantile(vals,[0,.25,.5,.75,.9,.95,.99,1]);assert np.array_equal(q,np.array(list(g['anchor_scores']['quantiles'].values())))
        checks['domain_arm_exact_summaries']+=1
    supplement=read(P/'MATCHED_NATIVE900_CASE.json');l=read(D/'SUPPLEMENTAL_INPUT_LOCK.json')
    assert sha(ROOT/'scripts/supplement_stvg_dino_feedback_native900_cache32_v1.py')==l['code_sha256']
    for f in l['native_files']:assert sha(ROOT/f['path'])==f['sha256']
    assert supplement['matched_queries']==1 and supplement['matched_frames']==4 and all(x['alias']=='H018'for x in supplement['qualifications'])
    native=load(ROOT/l['native_files'][0]['path'])
    for row in supplement['rows']:
        sc=np.asarray(native['raw'][row['position']]['detection']['target_all_scores'])
        assert row['original_selected_slot_score']==float(sc[row['selected_slot']]);checks['full900_selected_slot_scores']+=1
    assert not torch.cuda.is_initialized()
    export(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),same32_only=True,new_model_calls=0,new_predictions=0,new_GT_scores=0,
        CUDA_initialized=False,weights_media_GT_annotations_not_read=True,original_models_parameters_thresholds_unchanged=True,
        conditional_scalar_refs=len(raw['anchors']),candidate_trajectories=len(raw['candidates']),
        source_code_and_input_pins_all_exact=True,no_causal_percentage_claim=True,threshold_bins_not_parameter_selection=True))
    print(dict(checks))


if __name__=='__main__':main()
