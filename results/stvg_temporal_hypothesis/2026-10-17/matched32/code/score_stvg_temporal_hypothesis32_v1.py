"""Postseal B comparison with immutable A, all32 signed/unknown endpoints."""
from pathlib import Path
import sys,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis32_v1 import BASE,A,read,write,sha

def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    assert read(BASE/'PRIMARY_ROOT_READBACK.json')['status']=='pass';bars=dict(primary=read(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json'));full=(BASE/'full/ROOT_READBACK.json').exists()
    if full:assert read(BASE/'full/ROOT_READBACK.json')['status']=='pass';bars['full']=read(BASE/'full/GLOBAL_PREDICTION_BARRIER.json')
    assert full or read(BASE/'SIGNAL_COMPLETION.json')['informative_count']==0
    for b in bars.values():
        assert b['count']==32 and b['status']=='complete'
        for f in b['files']:assert sha(ROOT/f['path'])==f['sha256']
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,full_ran=full,GT_read=False));rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();maps={k:{f['case_id']:f for f in b['files']} for k,b in bars.items()};fits={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};old=read(A/'POSTSEAL_RESULTS.json');rows=list(old['rows']);usage=[];p_native={r['case_id']:r for r in old['rows'] if r['arm']=='Native'};check=0
    def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
    for r in rr:
        cid=r['probe_alias'];c=cases[cid];p=load(maps['primary'][cid]);preds=p['arms'];f=load(fits[cid])['fits']['HYP'];j=read(ROOT/jm[cid]['path']);d=read(BASE/'teacher'/(cid+'.json'));support=set()
        if full:
            preds['HYP_FULL_DIRECT_OPD']=load(maps['full'][cid])['arm'];support={a['position'] for row in read(BASE/'full/REFERENCE_LOCK.json')['records'] if row['case_id']==cid for a in row['anchors']}
        fps=float(c['input']['fps']);a,b=map(float,targets[cid]['span'])
        for name,pred in preds.items():
            v,_=metric._score_student_arm(pred,c,targets[cid],support,off,ind);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));cv,_=metric._score_student_arm(dict(boxes=pred['boxes'],interval=c['Native_interval']),c,targets[cid],support,off,ind);v['m_sIoU_common_Native_clock_GT']=cv['m_sIoU_dense_GT']
            if name=='Native':
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_common_Native_clock_GT']:assert v[k]==p_native[cid]['metrics'][k];check+=1
                continue
            if name!='HYP_FULL_DIRECT_OPD':assert np.array_equal(pred['boxes'],c['native_boxes_xyxy_normalized']);assert cv['m_sIoU_dense_GT']==p_native[cid]['metrics']['m_sIoU_common_Native_clock_GT']
            else:assert pred['interval']==preds['HYP_STUDENT']['interval']
            iv=pred['interval'];rows.append(dict(case_id=cid,dataset=r['dataset'],arm=name,metrics=v,boundary_metrics=dict(start_abs_error_seconds=abs(iv[0]-a)/fps,end_abs_error_seconds=abs(iv[1]-b)/fps,mean_abs_boundary_error_seconds=(abs(iv[0]-a)+abs(iv[1]-b))/(2*fps))))
        usage.append(dict(case_id=cid,dataset=r['dataset'],selection=j['selection']['status'],actual_distinct_observations=j['selection']['status']=='two_distinct_observations',informative=d['informative'],reward_difference=None if len(d['evidence'])!=2 or any(e['reward'] is None for e in d['evidence']) else d['evidence'][1]['reward']-d['evidence'][0]['reward'],rewards=f['rewards'],Native_conditional=list(f['Native_conditional']),Teacher_conditional=list(f['Teacher_conditional']),optimizer_steps=f['optimizer_steps'],no_information=f['no_information'],termination=f['termination']));del c,p,f;gc.collect()
    summary,paired=subset_aggregate(rows);write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,usage=usage,prelocked_gate=read(BASE/'DEVELOPMENT_GATE.json'),baseline_scalar_checks=check,GT_bindings=bindings,full_ran=full,GT_read=True,all_signed_unknown_retained=True,no_Candidate_or_Prompt_tuning=True,CURRENT_changed=False,cohort='same exposed development32, not independent confirmation'));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('NATIVE_HYPOTHESIS32_SCORED',flush=True)
if __name__=='__main__':main()
