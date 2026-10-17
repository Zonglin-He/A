"""All256 roots precede GT. Frozen/T2 exact, old32/new224 reported apart."""
from pathlib import Path
import copy,gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis256_v1 import BASE,OLD,T2,read,write,sha,verify
def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
    verify();assert read(BASE/'PRIMARY_ROOT_READBACK.json')['count']==256;bar=read(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==256 and bar['status']=='complete'
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=256,all_predictions_sealed_before_new_GT=True,GT_read=False));rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();pm={f['case_id']:f for f in bar['files']};tm={f['case_id']:f for f in read(T2/'GLOBAL_PREDICTION_BARRIER.json')['files']};fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};jm={f['case_id']:f for f in read(BASE/'INPUT_BARRIER.json')['jobs']};teachers={f['case_id']:f for f in read(BASE/'TEACHER_BARRIER.json')['files']};controls=validate_barrier('A');baseline={(r['case_id'],r['arm']):r['metrics'] for r in read(T2/'POSTSEAL_RESULTS.json')['rows']};old={(r['case_id'],r['arm']):r['metrics'] for r in read(OLD/'POSTSEAL_RESULTS.json')['rows']};rows=[];usage=[];checks=0
    def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
    for r in rr:
        cid=r['probe_alias'];c=cases[cid];p=load(pm[cid]);t2=load(tm[cid]);ctrl=load(controls[cid]);f=load(fm[cid])['fits']['HYP'];j=read(ROOT/jm[cid]['path']);d=read(ROOT/teachers[cid]['path']);preds=dict(Frozen=ctrl['arms']['F'],T2=copy.deepcopy(ctrl['arms']['F']),B=p['arms']['HYP_STUDENT']);preds['T2']['interval']=list(t2['selected_interval']);support=set()
        for name,pred in preds.items():
            assert np.array_equal(pred['boxes'],ctrl['arms']['F']['boxes']);v,_=metric._score_student_arm(pred,c,targets[cid],support,official,independent);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));cv,_=metric._score_student_arm(dict(boxes=pred['boxes'],interval=c['Native_interval']),c,targets[cid],support,official,independent);v['m_sIoU_common_Native_clock_GT']=cv['m_sIoU_dense_GT']
            if name in ['Frozen','T2']:
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5']:assert v[k]==baseline[cid,name][k],(cid,name,k);checks+=1
            if jm[cid]['reused_prior32']:
                a={'Frozen':'Native','T2':'T2','B':'HYP_STUDENT'}[name]
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_common_Native_clock_GT']:assert v[k]==old[cid,a][k];checks+=1
            rows.append(dict(case_id=cid,dataset=r['dataset'],arm=name,metrics=v))
        usage.append(dict(case_id=cid,dataset=r['dataset'],reused_prior32=jm[cid]['reused_prior32'],selection=j['selection']['status'],informative=d['informative'],rewards=f['rewards'],Native_conditional=list(f['Native_conditional']),Teacher_conditional=list(f['Teacher_conditional']),optimizer_steps=f['optimizer_steps'],termination=f['termination'],no_information=f['no_information'],final_interval_changed=f['final_interval']!=list(c['Native_interval'])));del c,p,t2,ctrl,f;gc.collect()
    summary,paired=subset_aggregate(rows);strata=[]
    for name,reuse in [('prior32',True),('added224',False)]:
        ids={u['case_id'] for u in usage if u['reused_prior32']==reuse};ss,pp=subset_aggregate([r for r in rows if r['case_id'] in ids]);strata.append(dict(name=name,count=len(ids),summary=ss,paired=pp))
    write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=256,per_domain=128,rows=rows,summary=summary,paired=paired,strata=strata,usage=usage,baseline_scalar_checks=checks,GT_bindings=bindings,GT_read=True,all_signed_unknown_retained=True,cohort='original core256 exposed development, not new confirmation',new_Spatial_fits=0,CURRENT_changed=False));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('NATIVE_HYPOTHESIS256_SCORED',flush=True)
if __name__=='__main__':main()
