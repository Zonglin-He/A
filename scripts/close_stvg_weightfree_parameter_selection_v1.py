"""Append corrected contrasts and freeze development-selected configurations."""
import sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_tuning_v1'
def main():
 from scripts import stvg_target_teacher_student_v1 as core
 sources=[('original',ROOT/'artifacts/stvg_weightfree_spatial_v1','diverse8'),('lr025',B,'lr025'),('lr050',B,'lr050')]+[(v,B/'factors002',v)for v in ['sigma050','sigma200','tau050','tau200','steps3','steps1']]+[(v,B/'interaction004',v)for v in ['combo1','combo2']]
 # Historical two scorers had a stale contrast-name list, while their rows/means were correct.
 for base in [B,B/'factors002']:
  rows=read(base/'POSTSEAL_ROWS.json')['rows'];lookup={(r['case_id'],r['arm']):r for r in rows};out=[]
  for ds in ['hc2','vidstg']:
   ids=sorted({r['case_id']for r in rows if r['dataset']==ds});assert len(ids)==4
   arms=sorted({r['arm']for r in rows if r['arm'].startswith('T_')})
   for arm in arms:
    for metric in rows[0]['metrics']:
     vals=[lookup[c,arm]['metrics'][metric]-lookup[c,'T']['metrics'][metric]for c in ids if lookup[c,arm]['metrics'][metric]is not None and lookup[c,'T']['metrics'][metric]is not None]
     out.append(dict(dataset=ds,left=arm,right='T',metric=metric,count=len(vals),mean_delta_pp=100*math.fsum(vals)/len(vals)if vals else None,positive=sum(v>1e-12 for v in vals),negative=sum(v< -1e-12 for v in vals),neutral=sum(abs(v)<=1e-12 for v in vals)))
  write(base/'CORRECTED_CONTRASTS002.json',dict(status='pass',GT_read=True,source_rows_sha256=sha(base/'POSTSEAL_ROWS.json'),contrasts=out,scope='correct stale arm-name dispatch only; original rows, summaries, scorer and negative evidence preserved; no new inference or GT scoring'))
 configs={};selected={};audit=[];evidence={}
 recipes=read(B/'interaction004/DESIGN_LOCK.json')['recipes']
 for ds in ['hc2','vidstg']:
  choices=[]
  for label,path,arm in sources:
   ss=read(path/'POSTSEAL_SUMMARY.json')['summary'];rr=[r for r in ss if r['dataset']==ds and r['arm']=='T_'+arm and r['metric']=='m_vIoU'];assert len(rr)==1 and rr[0]['count']==4
   cfg=core.load_original_opd_config(ds);cfg['steps']=5
   if label.startswith('lr'):cfg['lr']*=.25 if label=='lr025'else .5
   elif label.startswith('sigma'):cfg['sigma']*=.5 if label=='sigma050'else 2
   elif label.startswith('tau'):cfg['tau']*=.5 if label=='tau050'else 2
   elif label.startswith('steps'):cfg['steps']=int(label[-1])
   elif label.startswith('combo'):
    for k,v in recipes[ds][label].items():
     if k=='steps_absolute':cfg['steps']=v
     else:cfg[k]*=v
   choices.append(dict(dataset=ds,variant=label,mean_vIoU=rr[0]['mean'],config=cfg));evidence[str((path/'POSTSEAL_SUMMARY.json').relative_to(ROOT))]=sha(path/'POSTSEAL_SUMMARY.json')
  best=max(x['mean_vIoU']for x in choices);winner=sorted([x for x in choices if best-x['mean_vIoU']<=1e-12],key=lambda x:(x['config']['lr'],x['config']['steps'],x['variant']))[0]
  selected[ds]=winner['variant'];configs[ds]=winner['config'];audit+=choices
 write(B/'DEVELOPMENT_SELECTION_AUDIT.json',dict(status='complete',GT_read=True,records=audit,evidence=evidence,selected=selected,scope='adaptive 11-recipe/domain selection on four historical development sources/domain, not confirmation'))
 write(B/'METHOD_SELECTION_LOCK.json',dict(status='locked',time=time.time(),configs=configs,selected=selected,selected_using_development_GT=True,confirmation_GT_read=False,GT_read=False,GT_read_scope='this inference configuration carrier does not load annotations; development selection used labels',development_audit_sha256=sha(B/'DEVELOPMENT_SELECTION_AUDIT.json'),confirmation_roster_sha256=sha(B/'CONFIRMATION_ROSTER_LOCK.json'),rule='max development mean T+S vIoU; tie1e-12 lower lr then fewer steps then name',CURRENT_promoted=False))
 print(selected,configs)
if __name__=='__main__':main()
