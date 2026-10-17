"""Only after new256 ROOT/seal: seven matched arms and exact saved baselines."""
from pathlib import Path
import gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_marginal_decode256_v1 import BASE,PREV,read,write,sha,verify
B0=ROOT/'artifacts/stvg_temporal_hypothesis256_v1';B1=ROOT/'artifacts/stvg_temporal_marginal_i2_256_v1';T2=ROOT/'artifacts/stvg_temporal_t2_vs_t4_v1'
def main():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    verify();assert read(BASE/'PRIMARY_ROOT_READBACK.json')['status']=='pass'and read(BASE/'PRIMARY_ROOT_READBACK.json')['count']==256
    bar=read(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==len(bar['files'])==256
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    ctrl=read(BASE/'NO_UPDATE_CONTROL_BARRIER.json');assert ctrl['count']==256
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=256,all_actual_head_and_no_update_predictions_sealed_before_GT=True,GT_read=False))
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();pm={f['case_id']:f for f in bar['files']};bm={f['case_id']:f for f in read(B0/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};tm={f['case_id']:f for f in read(T2/'GLOBAL_PREDICTION_BARRIER.json')['files']};fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};jm={f['case_id']:f for f in read(BASE/'INPUT_BARRIER.json')['jobs']};teacher={f['case_id']:f for f in read(BASE/'TEACHER_BARRIER.json')['files']};baseline={(r['case_id'],r['arm']):r['metrics']for r in read(PREV/'POSTSEAL_RESULTS.json')['rows']};b1m={f['case_id']:f for f in read(B1/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};b2m={f['case_id']:f for f in read(PREV/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};diagnostics={r['case_id']:r for r in read(BASE/'DECODER_READBACK.json')['records']};rows=[];usage=[];checks=0
    def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
    for r in rr:
        cid=r['probe_alias'];c=cases[cid];p=load(pm[cid]);b=load(bm[cid]);b1=load(b1m[cid]);b2=load(b2m[cid]);t=load(tm[cid]);fit=load(fm[cid])['fits']['HYP'];j=read(ROOT/jm[cid]['path']);e=read(ROOT/teacher[cid]['path']);frozen=dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']),interval=list(c['Native_interval']));preds=dict(Frozen=frozen,T2=dict(frozen,interval=t['selected_interval']),B=b['arms']['HYP_STUDENT'],MARGINAL_NO_UPDATE=p['arms']['MARGINAL_NO_UPDATE'],B_MARGINAL_I2=b1['arms']['HYP_STUDENT'],B_MARGINAL_TRAIN=b2['arms']['HYP_STUDENT'],B_MARGINAL_DECODE=p['arms']['B_MARGINAL_DECODE'])
        for arm,pred in preds.items():
            assert np.array_equal(pred['boxes'],frozen['boxes']);m,_=metric._score_student_arm(pred,c,targets[cid],set(),official,independent);m.update(R_vIoU_at_0p3=float(m['m_vIoU']>=.3),R_vIoU_at_0p5=float(m['m_vIoU']>=.5));z,_=metric._score_student_arm(frozen,c,targets[cid],set(),official,independent);m['m_sIoU_common_Native_clock_GT']=z['m_sIoU_dense_GT']
            if arm!='B_MARGINAL_DECODE':
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_common_Native_clock_GT','R_vIoU_at_0p3','R_vIoU_at_0p5']:assert m[k]==baseline[cid,arm][k],(cid,arm,k);checks+=1
            rows.append(dict(case_id=cid,dataset=r['dataset'],arm=arm,metrics=m))
        hs=j['selection']['hypotheses'];usage.append(dict(case_id=cid,dataset=r['dataset'],I1_I2_and_Teacher_exact_reuse=True,Native_interval_probabilities=list(fit['Native_interval_probabilities']),informative=e['informative'],rewards=fit['rewards'],Native_conditional=list(fit['Native_conditional']),Teacher_conditional=list(fit['Teacher_conditional']),optimizer_steps=fit['optimizer_steps'],termination=fit['termination'],no_information=fit['no_information'],**{k:v for k,v in diagnostics[cid].items()if k not in ['case_id','dataset','no_information']}));del c,p,b,b1,b2,t,fit;gc.collect()
    summary,paired=subset_aggregate(rows);write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=256,per_domain=128,rows=rows,summary=summary,paired=paired,usage=usage,baseline_scalar_checks=checks,GT_bindings=bindings,GT_read=True,all_signed_unknown_retained=True,cohort='same exposed original core256',only_final_decoder_changed=True,new_fits=0,new_GPU_forwards=0,new_Teacher_calls=0,reused_caption_and_support_pairs=512,new_Spatial_fits=0,CURRENT_changed=False));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));assert checks==9216;print('MARGINAL_DECODE_256_SCORED',flush=True)
if __name__=='__main__':main()
