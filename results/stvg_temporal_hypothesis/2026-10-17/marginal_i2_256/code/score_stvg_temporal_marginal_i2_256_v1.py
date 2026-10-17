"""Only after new256 ROOT/seal: five matched arms and exact saved baselines."""
from pathlib import Path
import gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_marginal_i2_256_v1 import BASE,OLD,T2,read,write,sha,verify
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
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();pm={f['case_id']:f for f in bar['files']};bm={f['case_id']:f for f in read(OLD/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};tm={f['case_id']:f for f in read(T2/'GLOBAL_PREDICTION_BARRIER.json')['files']};fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};jm={f['case_id']:f for f in read(BASE/'INPUT_BARRIER.json')['jobs']};teacher={f['case_id']:f for f in read(BASE/'TEACHER_BARRIER.json')['files']};baseline={(r['case_id'],r['arm']):r['metrics']for r in read(OLD/'POSTSEAL_RESULTS.json')['rows']};probe={(r['case_id'],r['arm']):r['metrics']for r in read(ROOT/'artifacts/stvg_temporal_b_diagnosis256_v1/native_marginal_probe001/RESULTS.json')['rows']};rows=[];usage=[];checks=0
    def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
    for r in rr:
        cid=r['probe_alias'];c=cases[cid];p=load(pm[cid]);b=load(bm[cid]);t=load(tm[cid]);fit=load(fm[cid])['fits']['HYP'];j=read(ROOT/jm[cid]['path']);e=read(ROOT/teacher[cid]['path']);frozen=dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']),interval=list(c['Native_interval']));preds=dict(Frozen=frozen,T2=dict(frozen,interval=t['selected_interval']),B=b['arms']['HYP_STUDENT'],MARGINAL_NO_UPDATE=p['arms']['MARGINAL_NO_UPDATE'],B_MARGINAL_I2=p['arms']['HYP_STUDENT'])
        for arm,pred in preds.items():
            assert np.array_equal(pred['boxes'],frozen['boxes']);m,_=metric._score_student_arm(pred,c,targets[cid],set(),official,independent);m.update(R_vIoU_at_0p3=float(m['m_vIoU']>=.3),R_vIoU_at_0p5=float(m['m_vIoU']>=.5));z,_=metric._score_student_arm(frozen,c,targets[cid],set(),official,independent);m['m_sIoU_common_Native_clock_GT']=z['m_sIoU_dense_GT']
            if arm in ['Frozen','T2','B']:
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_common_Native_clock_GT','R_vIoU_at_0p3','R_vIoU_at_0p5']:assert m[k]==baseline[cid,arm][k],(cid,arm,k);checks+=1
            if arm=='MARGINAL_NO_UPDATE':
                for k in ['m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_common_Native_clock_GT','R_vIoU_at_0p3','R_vIoU_at_0p5']:assert m[k]==probe[cid,'Native_marginal_MAP_CPU'][k],(cid,arm,k);checks+=1
            rows.append(dict(case_id=cid,dataset=r['dataset'],arm=arm,metrics=m))
        hs=j['selection']['hypotheses'];usage.append(dict(case_id=cid,dataset=r['dataset'],I2_changed=not next(x for x in read(BASE/'INPUT_CHANGE_READBACK.json')['records']if x['case_id']==cid)['I2_identical_to_old'],reused_slots=[x['slot']for x in j['evidence_reuse']],informative=e['informative'],rewards=fit['rewards'],Native_conditional=list(fit['Native_conditional']),Teacher_conditional=list(fit['Teacher_conditional']),optimizer_steps=fit['optimizer_steps'],termination=fit['termination'],no_information=fit['no_information'],final_interval_changed=fit['final_interval']!=list(c['Native_interval']),final_equals_target_hard=fit['final_interval']==(list(c['Native_interval'])if fit['no_information']else hs[int(np.argmax(fit['Teacher_conditional']))]['interval_frames'])));del c,p,b,t,fit;gc.collect()
    summary,paired=subset_aggregate(rows);write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=256,per_domain=128,rows=rows,summary=summary,paired=paired,usage=usage,baseline_scalar_checks=checks,GT_bindings=bindings,GT_read=True,all_signed_unknown_retained=True,cohort='same exposed original core256',only_I2_ranking_changed=True,new_Spatial_fits=0,CURRENT_changed=False));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('MARGINAL_I2_256_SCORED',flush=True)
if __name__=='__main__':main()
