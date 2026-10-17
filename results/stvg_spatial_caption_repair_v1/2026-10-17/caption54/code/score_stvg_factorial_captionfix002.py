"""Postseal Native factorial with caption-matched same-clock Full-B repair.

All35 affected core queries fell back to the Native Temporal interval. Exact
reuse requires identical original E1 jobs, Source, temporal output and Student
recipe. A complete executed Native-B prediction is then also the corrected
Adapted-clock Full-B prediction; no frame/window splicing or GT-based routing.
"""
from pathlib import Path
import inspect,sys,copy,json,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import BASE,FIXED,CORE,read,put,sha,verify
def main():
    import torch,numpy as np
    from scripts import score_stvg_factorial_native256_v1 as original
    from scripts.stvg_overnight_score_v1 import seal,aggregate
    from scripts.stvg_overnight_common_v1 import score as old_score
    verify();missing={r['case_id'] for r in read(BASE/'DIAGNOSIS.json')['records'] if r['cohort']=='core256'};assert len(missing)==35
    jobs={(j['case_id'],j['center_slot']):j for j in read(FIXED/'TEACHER_JOB_LOCK.json')['records']};oldjobs={(j['case_id'],j['center_slot']):j for j in read(CORE/'TEACHER_JOB_LOCK.json')['records']};tt={r['case_id']:r['selected_interval'] for r in read(CORE/'TEMPORAL_LOCK.json')['records']};native={r['case_id']:r['selected_interval'] for r in read(FIXED/'TEMPORAL_LOCK.json')['records']}
    for cid in missing:
        assert tt[cid]==native[cid]
        for slot in range(4):assert jobs[cid,slot]==oldjobs[cid,slot]
    for arm in ('A','B'):
        seal(FIXED/'arms'/arm);new=read(FIXED/'arms'/arm/'trials/G32/DESIGN_LOCK.json');old=read(CORE/'trials'/arm/'DESIGN_LOCK.json')
        for k in ('configs','response_settings','recipe'):assert new[k]==old[k]
    def matched_repair(cid,arm,current,old):
        if arm!='B' or cid not in missing:return old
        assert current['temporal_record']==old['temporal_record'] and current['arms']['T']['interval']==native[cid]
        out=copy.deepcopy(old)
        for key in ('S_M2','TS_M2'):out['arms'][key]=copy.deepcopy(current['arms'][key])
        return out
    src=inspect.getsource(original.main);needle="            for k in ('F','T'):";assert src.count(needle)==1;src=src.replace(needle,"            old=matched_repair(cid,arm,d,old)\n"+needle)
    ns=dict(original.__dict__);ns.update(BASE=FIXED,CORE=CORE,verify=verify,matched_repair=matched_repair);exec(compile(src,__file__,'exec'),ns);ns['main']()
    d=read(FIXED/'POSTSEAL_RESULTS.json');historical=old_score('B');rows=[]
    for r in historical['rows']:
        if r['arm']!='B':continue
        vals=dict(r['metrics']);vals.update(R_vIoU_at_0p3=float(vals['m_vIoU']>=.3),R_vIoU_at_0p5=float(vals['m_vIoU']>=.5));rows.append(dict(case_id=r['case_id'],dataset=r['dataset'],arm='Full_B_original_missingCaption',metrics=vals))
    d['rows']+=rows;d['summary'],d['paired']=aggregate(d['rows'],256);d.update(Full_B_Adapted_definition='original221 plus35 complete actual Native-B outputs with independently identical E1 jobs, Source, final T interval, recipe and episodic seed; caption-input repair only',historical_Full_B_preserved_in_rows=True,confirmation128_not_rescored=True,caption_repair_new_Temporal_fits=0)
    put(FIXED/'POSTSEAL_RESULTS.json',d,replace=True);put(FIXED/'CPU_COMPLETION.json',dict(status='complete',count=256,GT_read=True,result_sha256=sha(FIXED/'POSTSEAL_RESULTS.json')),replace=True);put(FIXED/'ROOT_CAPTION_MATCHED_FULL_REUSE.json',dict(status='pass',count=35,complete_query_predictions_only=True,no_frame_window_or_state_splicing=True,identical_Native_and_existing_T_intervals=True,identical_original_E1_jobs_and_source_recipes=True,predetermined_Caption_missing_status_not_GT_selection=True,original_B_results_preserved=True,GT_read=False));print('CAPTION_MATCHED_FACTORIAL256_POSTSEAL_COMPLETE')
if __name__=='__main__':main()
