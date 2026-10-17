"""Independent postseal temporal comparison, fixed-spatial and new-time Full."""
from pathlib import Path
import gc,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_semantic_view32_v1 import BASE,read,write,sha
FULL=BASE/'full'
def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    assert read(BASE/'PRIMARY_ROOT_READBACK.json')['status']=='pass' and read(FULL/'ROOT_READBACK.json')['status']=='pass'
    bars=dict(primary=read(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json'),full=read(FULL/'GLOBAL_PREDICTION_BARRIER.json'),old=read(ROOT/'artifacts/stvg_r1_direct_reliable32_v1/arms/R1_FIXED_OPD/trials/G32/GLOBAL_PREDICTION_BARRIER.json'))
    assert all(b['count']==32 and b['status']=='complete' for b in bars.values())
    for b in bars.values():
        for f in b['files']:assert sha(ROOT/f['path'])==f['sha256']
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,all_new_temporal_and_full_sealed_before_GT=True,GT_read=False))
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();maps={k:{f['case_id']:f for f in b['files']} for k,b in bars.items()};bm={f['case_id']:f for f in read(BASE/'BASELINE_BARRIER.json')['files']};jm={f['case_id']:f for f in read(BASE/'INPUT_BARRIER.json')['jobs']};fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};tm={f['case_id']:f for f in read(BASE/'TARGET_BARRIER.json')['files']};refs={r['case_id']:r['anchors'] for r in read(FULL/'REFERENCE_LOCK.json')['records']};rows=[];usage=[];strata=[];invariants=[]
    oldscalar={(r['case_id'],r['arm']):r['metrics'] for r in read(ROOT/'artifacts/stvg_r1_direct_reliable32_v1/POSTSEAL_RESULTS.json')['rows']}
    def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
    for r in rr:
        cid=r['probe_alias'];c=cases[cid];p=load(maps['primary'][cid]);preds=p['arms'];old=load(bm[cid]);assert np.array_equal(preds['Native']['boxes'],old['arms']['F']['boxes']);preds['T2']=old['arms']['T'];preds['SEM_FULL_DIRECT_OPD']=load(maps['full'][cid])['arm'];preds['T2_R1_DIRECT_OPD']=load(maps['old'][cid])['arms']['TS_M2'];ff=load(fm[cid])['fits'];td=load(tm[cid]);j=read(ROOT/jm[cid]['path']);teacher=read(BASE/'teacher'/(cid+'.json'));fps=j['source_fps'];a,b=map(float,targets[cid]['span']);gaps=np.diff(j['sampled_physical_frame_ids']);stride=float(np.median(gaps));sg=dict(case_id=cid,dataset=r['dataset'],shorter_than_uniform8_median_gap=(b-a)<stride,boundary_between_observed_frames=any(float(x) not in j['sampled_physical_frame_ids'] for x in [a,b-1]),role_swap_or_nonexistent_GT_label='unknown');strata.append(sg)
        support={a['position'] for a in refs[cid]};native_s=None
        for name,pred in preds.items():
            v,_=metric._score_student_arm(pred,c,targets[cid],support,official,independent);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));iv=pred['interval'];boundary=dict(start_abs_error_seconds=abs(iv[0]-a)/fps,end_abs_error_seconds=abs(iv[1]-b)/fps,mean_abs_boundary_error_seconds=(abs(iv[0]-a)+abs(iv[1]-b))/(2*fps))
            if name not in ['SEM_FULL_DIRECT_OPD','T2_R1_DIRECT_OPD']:assert np.array_equal(pred['boxes'],preds['Native']['boxes'])
            if name=='SEM_FULL_DIRECT_OPD':assert pred['interval']==preds['SEM_STUDENT']['interval']
            if name=='Native':native_s=v['m_sIoU_dense_GT']
            elif name not in ['SEM_FULL_DIRECT_OPD','T2_R1_DIRECT_OPD']:assert v['m_sIoU_dense_GT']==native_s
            if name in ['Native','T2','T2_R1_DIRECT_OPD']:
                oldname={'Native':'Frozen','T2':'T2','T2_R1_DIRECT_OPD':'R1_FIXED_OPD'}[name]
                # Support/unsupervised sets differ; dense metrics must be exact.
                for key in ['m_vIoU','m_tIoU','m_sIoU_dense_GT']:assert v[key]==oldscalar[cid,oldname][key]
            rows.append(dict(case_id=cid,dataset=r['dataset'],arm=name,metrics=v,boundary_metrics=boundary))
        for arm in ['SEM','DIRECT','PARA']:usage.append(dict(case_id=cid,dataset=r['dataset'],arm=arm,admitted=teacher['admitted'][arm],optimizer_steps=ff[arm]['optimizer_steps'],no_information=ff[arm]['no_information'],termination=ff[arm]['termination'],Teacher_to_Native_KL=td['Teacher_to_Native_KL'][arm],caption_check=teacher['caption_check']['label'],boundary_parse=teacher['Direct_boundary']['parsed']['status'],paraphrase_check=teacher['paraphrase_check']['label']))
        invariants.append(dict(case_id=cid,all_temporal_only_spatial_boxes_exact=True,Full_tIoU_equals_semantic_T_only=True,GT_read=True));del old,p,td,ff;gc.collect()
    summary,paired=subset_aggregate(rows)
    boundary_rows=[dict(case_id=r['case_id'],dataset=r['dataset'],arm=r['arm'],metrics=r['boundary_metrics']) for r in rows];bs,bp=subset_aggregate(boundary_rows)
    # Boundary outputs are seconds; helper's conventional x100 field is not pp.
    for p in bp:
        p['mean_delta_seconds']=p.pop('mean_delta_pp')/100
        p['paired_source_t95_interval_seconds']=[v/100 for v in p.pop('paired_source_t95_interval_pp')] if p['valid']>1 else None
        p['per_source_delta_seconds']={k:v/100 for k,v in p.pop('per_source_delta_pp').items()}
    subs=[]
    for key in ['shorter_than_uniform8_median_gap','boundary_between_observed_frames']:
        ids={r['case_id'] for r in strata if r[key]};sr=[r for r in rows if r['case_id'] in ids];ss,sp=subset_aggregate(sr);subs.append(dict(stratum=key,count=len(ids),summary=ss,paired=sp))
    common={r['probe_alias'] for r in rr if all(x['admitted'] for x in usage if x['case_id']==r['probe_alias'])};cr=[r for r in rows if r['case_id'] in common];cs,cp=subset_aggregate(cr)
    write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,boundary_summary=bs,boundary_paired=bp,usage=usage,strata=strata,subgroup_results=subs,common_admission=dict(count=len(common),summary=cs,paired=cp),invariants=invariants,GT_bindings=bindings,GT_read=True,all_signed_unknown_retained=True,cohort='same exposed development32, not independent confirmation',CURRENT_changed=False))
    write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('TEMPORAL_SEMANTIC32_SCORED',flush=True)
if __name__=='__main__':main()
