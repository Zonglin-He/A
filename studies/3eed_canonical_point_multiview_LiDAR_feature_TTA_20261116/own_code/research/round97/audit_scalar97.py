"""Separate full-list reducer, negative tails and no-update controls are never filtered."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round97';B=R/'crossplatform/round97';S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 expected={'waymo':3782,'drone':2792,'quad':2736};summ=rd(S/'ALL_METHOD_ROWS.json');agg=rd(S/'AGGREGATE_ROWS.json');pairs=rd(S/'ALL_PAIRED_ROWS.json');n=0;readouts=0;updates=[]
 for src in expected:
  for tgt in expected:
   if src==tgt:continue
   rr=rd(S/(src+'_to_'+tgt+'_ALL_ROWS_PRIVATE.json'));assert len(rr)==expected[tgt];n+=len(rr)
   for arm in rr[0]['arms']:
    x=np.array([r['arms'][arm]['IoU'] for r in rr]);assert ((x>=0)&(x<=1+1e-12)).all();s=next(s for s in summ if s['source']==src and s['target']==tgt and s['arm']==arm);assert s['n']==len(x)
    for k,v in [('mIoU_percent',x.mean()*100),('Acc25_percent',(x>.25).mean()*100),('Acc50_percent',(x>.5).mean()*100)]:assert abs(s[k]-v)<1e-10
    readouts+=len(x)
   for r in rr:
    for arm in ['physical_raw','physical_bt','wrong_physical_raw','wrong_physical_bt']:
     rec=r['updates'][arm];assert len(rec['trials'])<=9
     if arm.endswith('_bt'):
      accepted=[t for t in rec['trials'] if t['accepted']]
      if accepted:assert rec['alpha']==accepted[0]['alpha'] and accepted[0]['objective']<=accepted[0]['bound']+1e-10
      else:assert rec['alpha']==0 and rec['actual_parameter_L2']==0 and r['arms'][arm]['query_PRIVATE']==r['arms']['prepared_frozen']['query_PRIVATE']
     updates.append({'source':src,'target':tgt,'arm':arm,'alpha':rec['alpha'],'nonzero':rec['actual_parameter_L2']>0,'trial_count':len(rec['trials'])})
   for p in [p for p in pairs if p['source']==src and p['target']==tgt]:
    d=np.array([r['arms']['physical_bt']['IoU']-r['arms'][p['control']]['IoU'] for r in rr]);assert abs(d.mean()*100-p['gain_pp'])<1e-10 and (d>1e-12).sum()==p['better'] and (d< -1e-12).sum()==p['worse'] and p['better']+p['worse']+p['same']==len(rr)
 for a in agg:
  ss=[s for s in summ if s['arm']==a['arm'] and (a['target_group']=='all_six_equal' or s['target']==a['target_group'])];assert len(ss)==a['directions']
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(a[k]-np.mean([s[k] for s in ss]))<1e-10
 assert n==18620 and readouts==148960 and len(summ)==48
 u=[]
 for src in expected:
  for tgt in expected:
   if src==tgt:continue
   for arm in ['physical_raw','physical_bt','wrong_physical_raw','wrong_physical_bt']:
    rr=[x for x in updates if x['source']==src and x['target']==tgt and x['arm']==arm];u.append({'source':src,'target':tgt,'arm':arm,'n':len(rr),'nonzero_updates':sum(r['nonzero'] for r in rr),'alpha_mean':float(np.mean([r['alpha'] for r in rr])),'trial_count_mean':float(np.mean([r['trial_count'] for r in rr]))})
 (H/'UPDATE_SUMMARY_ROWS.json').write_text(json.dumps(u,indent=2)+'\n');out={'state':'PASS_R97_INDEPENDENT_ALL_READOUT_METRICS_PAIRED_NEGATIVES_ACCEPTANCE_AND_PLATFORM_REDUCTION','directions':6,'n':n,'readouts':readouts,'summary_rows':len(summ),'all_negatives_retained':True,'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
