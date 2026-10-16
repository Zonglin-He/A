"""Whole-study arithmetic readback from three already sealed group results."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE,read,put,sha
from scripts.score_stvg_referent_abc_v1 import aggregate
def main():
    groups={};rows={};sem={};usage=[];bindings={}
    for arm in ('A','B','C'):
        b=BASE/'trials'/arm;d=read(b/'POSTSEAL_RESULTS.json');assert read(b/'CPU_COMPLETION.json')['result_sha256']==sha(b/'POSTSEAL_RESULTS.json');s,p=aggregate(d['rows']);assert s==d['summary']and p==d['paired'];assert read(b/'ROOT_READBACK.json')['status']=='pass';bindings[arm]=dict(result_sha256=sha(b/'POSTSEAL_RESULTS.json'),root_sha256=sha(b/'ROOT_READBACK.json'),prediction_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'));groups[arm]=d
        for r in d['rows']:
            key=r['case_id'],r['arm']
            if key in rows:assert rows[key]==r
            rows[key]=r
        for r in d['semantic']['rows']:
            key=r['case_id'],r['slot'],r['arm']
            if key in sem:assert sem[key]==r
            sem[key]=r
        usage += [dict(x,arm=arm)for x in d['usage']]
    rows=list(rows.values());assert len(rows)==160;summary,paired=aggregate(rows);changes=[]
    for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
        results={a:read(BASE/'teacher'/a/(j['job_id']+'.json'))['parsed']for a in ('A','B','C')};pol=read(BASE/'native_cache'/j['job_id']/'POLICIES.json');changes.append(dict(case_id=j['case_id'],dataset=j['dataset'],slot=j['center_slot'],selections={a:dict(status=x['status'],visible_id=x.get('visible_id'))for a,x in results.items()},B_C_same_choice=results['B'].get('visible_id')==results['C'].get('visible_id')and results['B']['status']==results['C']['status'],B_C_transition_changed=any(x['transition_changed']for r in pol['records']for x in r['comparability']),B_C_common_fixed_and_object_indices_verified=all(x['B_C_fixed_indices_equal']and x['B_C_object_indices_equal']for r in pol['records']for x in r['comparability'])))
    chains=[];idx={(r['case_id'],r['arm']):r['metrics']for r in rows}
    for cid in sorted({r['case_id']for r in sem.values()}):
        for arm in ('A','B','C'):
            r=[r for r in sem.values()if r['case_id']==cid and r['arm']==arm];u=next(x for x in usage if x['case_id']==cid and x['arm']==arm);t=idx[cid,'Temporal']['m_vIoU'];a=idx[cid,'A']['m_vIoU'];v=idx[cid,arm]['m_vIoU'];chains.append(dict(case_id=cid,dataset=r[0]['dataset'],arm=arm,semantic_windows=[{k:x[k]for k in ('slot','identity_known','selected_visible_id','category','role_status')}for x in r],anchors=u['anchors'],actual_fit=u['actual_fit'],accepted_steps=u['accepted_steps'],nonzero_final_update=u['nonzero_parameter_update'],Student_vIoU=v,Student_minus_Temporal_pp=100*(v-t),Student_minus_A_pp=100*(v-a),posthoc_readout_only=True))
    put(BASE/'STUDY_READBACK.json',dict(status='pass',count=16,row_count=160,groups=bindings,rows=rows,summary=summary,paired=paired,semantic_rows=list(sem.values()),usage=usage,selection_changes=changes,semantic_update_chains=chains,GT_read=True,no_new_GT_scoring=True,all_three_groups_and_all_unknowns_preserved=True,source_count_not_48=True,scope='fixed exposed development only; no domain-gain gate or automatic larger experiment'))
    print(json.dumps(dict(status='pass',row_count=160,comparisons=[r for r in paired if r['metric']=='m_vIoU'and r['comparison']in ('B_minus_A','C_minus_B','C_minus_A','A_minus_Temporal','B_minus_Temporal','C_minus_Temporal')])) )
if __name__=='__main__':main()
