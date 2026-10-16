"""Postseal twelve-source paired control; no candidate or parameter selection."""
from pathlib import Path
import collections,gc,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_geometry_audit_v1 import BASE as STUDY,LITE,read,put,sha
BASE=STUDY/'clock_control'

def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    pred=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert pred['status']=='complete'and pred['count']==12 and pred['GT_read']is False
    assert pred['root_sha256']==sha(BASE/'trials/C_CLOCK/ROOT_READBACK.json')and pred['teacher_root_sha256']==sha(BASE/'ROOT_TEACHER_READBACK.json')
    roster=read(STUDY/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in roster};newfiles={r['case_id']:r for r in pred['files']};oldfiles={r['case_id']:r for r in read(LITE/'trials/C_FULL/STUDENT_GLOBAL_BARRIER.json')['files']}
    # Entire corrected-control prediction group and independent saved-math root
    # have been checked before this stage's first raw annotation access.
    targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions()
    rows=[]
    for rec in roster:
        cid=rec['probe_alias'];case=cases[cid];oldentry=oldfiles[cid];newentry=newfiles[cid];assert sha(ROOT/oldentry['path'])==oldentry['sha256']and sha(ROOT/newentry['path'])==newentry['sha256']
        old=torch.load(ROOT/oldentry['path'],map_location='cpu',weights_only=False);new=torch.load(ROOT/newentry['path'],map_location='cpu',weights_only=False)
        assert old['source_case_sha256']==new['source_case_sha256']==case['source_case_sha256']and old['arms']['T']['interval']==new['arms']['TS_M2']['interval']
        pool=read(LITE/'reclocked_inputs/new_crop/proposals'/f'{cid}.json');support=[w['source_grid_position']for w in pool['windows']]
        arms={'Frozen':old['arms']['F'],'Temporal_Lite':old['arms']['T'],'Legacy_Full':old['arms']['TS_M2'],'Corrected_Spatial_Clock_Full':new['arms']['TS_M2'],'Legacy_Matched_OPD':old['arms']['TS_M0']}
        for arm,data in arms.items():
            metrics,audit=metric._score_student_arm(data,case,targets[cid],support,official,independent)
            rows.append(dict(case_id=cid,dataset=rec['dataset'],stratum=rec['diagnostic_stratum'],arm=arm,metrics=metrics,real_new_decode=newentry['new_fit']if arm=='Corrected_Spatial_Clock_Full'else False,GT_read=True))
        del old,new;gc.collect()
    summary=[];paired=[]
    for ds in ('hc2','vidstg'):
        for arm in sorted({r['arm']for r in rows}):
            sub=[r for r in rows if r['dataset']==ds and r['arm']==arm]
            for name in sub[0]['metrics']:
                vals=[r['metrics'][name]for r in sub if r['metrics'][name]is not None]
                summary.append(dict(dataset=ds,arm=arm,metric=name,count=len(sub),valid=len(vals),unknown=len(sub)-len(vals),mean=math.fsum(vals)/len(vals)if vals else None))
        by={(r['case_id'],r['arm']):r for r in rows if r['dataset']==ds}
        for before in ('Frozen','Temporal_Lite','Legacy_Full'):
            vals={cid:100*(by[cid,'Corrected_Spatial_Clock_Full']['metrics']['m_vIoU']-by[cid,before]['metrics']['m_vIoU'])for cid in cases if(cid,before)in by}
            paired.append(dict(dataset=ds,comparison='Corrected_Spatial_Clock_Full_minus_'+before,metric='m_vIoU',count=len(vals),mean_delta_pp=math.fsum(vals.values())/len(vals),positive=sum(v>1e-8 for v in vals.values()),negative=sum(v< -1e-8 for v in vals.values()),neutral=sum(abs(v)<=1e-8 for v in vals.values()),per_source_delta_pp=vals))
    put(BASE/'POSTSEAL_SCORE.json',dict(status='complete',count=12,rows=rows,summary=summary,paired=paired,GT_read=True,prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),GT_bindings=bindings,common_support='the same four original physical observation positions for every arm; final full Tube unchanged official metric',selection_scope='stratified exposed development diagnosis only; no confirmation claim',legacy_M0_scope='same historical parameters and original reference only, not a new corrected-reference M0 run'))
    put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=12,row_count=len(rows),GT_read=True,score_sha256=sha(BASE/'POSTSEAL_SCORE.json'),new_fit_count=1,exact_unchanged_reference_reuse_count=11))
    print(json.dumps({'status':'complete','paired':paired}))

if __name__=='__main__':main()
