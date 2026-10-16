"""Final matched prompt interaction; no new models, GT or recipe selection."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_spatial_who_order_v1 import BASE,ABC,read,put,sha
from scripts.score_stvg_referent_abc_v1 import aggregate
def main():
    rows={};semantic={};usage=[];geometry=[];bindings={};teacher=[]
    for a in ('FULL_ORDER','A_ORDER'):
        b=BASE/'trials'/a;d=read(b/'POSTSEAL_RESULTS_EXTENDED.json');assert read(b/'EXTENDED_CPU_COMPLETION.json')['result_sha256']==sha(b/'POSTSEAL_RESULTS_EXTENDED.json');s,p=aggregate(d['rows']);assert s==d['summary']and p==d['paired'];assert read(b/'ROOT_READBACK.json')['status']=='pass'
        for r in d['rows']:
            k=r['case_id'],r['arm'];assert k not in rows or rows[k]==r;rows[k]=r
        for r in d['semantic']['rows']:
            k=r['case_id'],r['slot'],r['arm'];assert k not in semantic or semantic[k]==r;semantic[k]=r
        usage += [dict(r,arm=a)for r in d['usage']];geometry += [dict(r,arm=a)for r in d['geometry']];bindings[a]=dict(Student_root_sha256=sha(b/'ROOT_READBACK.json'),result_sha256=sha(b/'POSTSEAL_RESULTS_EXTENDED.json'),Teacher_root_sha256=sha(BASE/'teacher'/a/'ROOT_READBACK.json'))
        for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
            d=read(BASE/'teacher'/a/(j['job_id']+'.json'));teacher.append(dict(case_id=j['case_id'],dataset=j['dataset'],arm=a,slot=j['center_slot'],parse_status=d['parsed']['status'],visible_id=d['parsed'].get('visible_id'),abstained=d['parsed'].get('visible_id')==0,truncated=d['truncated'],GT_read=False))
    rows=list(rows.values());s,p=aggregate(rows);ix={(r['case_id'],r['arm']):r['metrics']for r in rows};interaction=[]
    for ds in ('hc2','vidstg'):
        ids=sorted({r['case_id']for r in rows if r['dataset']==ds})
        for m in ix[ids[0],'A']:
            v={cid:100*(ix[cid,'A_ORDER'][m]-ix[cid,'A'][m]-ix[cid,'FULL_ORDER'][m]+ix[cid,'Full_E1'][m])for cid in ids if all(ix[cid,a][m]is not None for a in ('A_ORDER','A','FULL_ORDER','Full_E1'))};interaction.append(dict(dataset=ds,metric=m,contrast='(A_ORDER-A)-(FULL_ORDER-Full_E1)',valid=len(v),unknown=8-len(v),mean_delta_pp=math.fsum(v.values())/len(v)if v else None,per_source_delta_pp=v))
    put(BASE/'STUDY_READBACK.json',dict(status='pass',count=16,rows=rows,summary=s,paired=p,interactions=interaction,semantic_rows=list(semantic.values()),usage=usage,geometry=geometry,Teacher_decisions=teacher,bindings=bindings,GT_read=True,all_source_positive_negative_neutral_unknown_preserved=True,no_new_GT_or_model_execution=True))
    print(json.dumps(dict(status='pass',summary=[r for r in s if r['metric']=='m_vIoU'],interactions=[r for r in interaction if r['metric']=='m_vIoU'])))
if __name__=='__main__':main()
