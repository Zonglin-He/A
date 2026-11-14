"""Independent all-input reducer and primary-vs-all-fixed-controls accounting."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round98';S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 summary=rd(S/'ALL_METHOD_ROWS.json');agg=rd(S/'AGGREGATE_ROWS.json');pairs=rd(S/'ALL_PAIRED_ROWS.json');total=values=0;fitrows=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   rows=rd(S/(src+'_to_'+tgt+'_ALL_ROWS_PRIVATE.json'));assert len(rows)=={'waymo':3782,'drone':2792,'quad':2736}[tgt];total+=len(rows)
   for arm in rows[0]['arms']:
    x=np.array([r['arms'][arm]['IoU'] for r in rows]);assert ((x>=0)&(x<=1+1e-12)).all();m=next(x for x in summary if x['source']==src and x['target']==tgt and x['arm']==arm);assert m['n']==len(rows)
    for k,v in [('mIoU_percent',x.mean()*100),('Acc25_percent',(x>.25).mean()*100),('Acc50_percent',(x>.5).mean()*100)]:assert abs(m[k]-v)<1e-10
    values+=len(rows)
   for p in [p for p in pairs if p['source']==src and p['target']==tgt]:
    a=np.array([r['arms']['lm_FC']['IoU']-r['arms'][p['control']]['IoU'] for r in rows]);assert abs(a.mean()*100-p['gain_pp'])<1e-10 and int((a>1e-12).sum())==p['better'] and int((a< -1e-12).sum())==p['worse'];assert p['better']+p['worse']+p['same']==len(rows)
   for r in rows:
    for name,u in r['fits'].items():assert u['final_error']<=u['initial_error']+1e-10 and u['accepted_steps']<=10 and u['attempted_steps']<=60
   for name in rows[0]['fits']:
    uu=[r['fits'][name] for r in rows];fitrows.append({'source':src,'target':tgt,'fit':name,'n':len(uu),'initial_error_mean':float(np.mean([u['initial_error'] for u in uu])),'final_error_mean':float(np.mean([u['final_error'] for u in uu])),'relative_error_mean':float(np.mean([u['relative_error'] for u in uu])),'accepted_steps_mean':float(np.mean([u['accepted_steps'] for u in uu])),'attempted_steps_sum':sum(u['attempted_steps'] for u in uu),'nonzero_actual_parameter_updates':sum(u['parameter_L2']>0 for u in uu)})
 for a in agg:
  ss=[x for x in summary if x['arm']==a['arm'] and (a['target_group']=='all_six_equal' or x['target']==a['target_group'])];assert len(ss)==a['directions']
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(a[k]-np.mean([x[k] for x in ss]))<1e-10
 assert total==18620 and values==539980 and len(summary)==174 and len(agg)==116
 (H/'FIT_SUMMARY_ROWS.json').write_text(json.dumps(fitrows,indent=2)+'\n');out={'state':'PASS_R98_ALL_SIX_TASK_METRICS_ALL_NEGATIVES_FIXED_CONTROLS_AND_MACRO_REDUCTION','n':total,'readouts':values,'summary_rows':len(summary),'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
