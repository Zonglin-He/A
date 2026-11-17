"""Pre-task-score descriptive evidence quality, not runtime filtering."""
from common146 import *
import numpy as np,collections

def main():
 check();assert rd(H/'IMAGE_AUDIT.json')['status']=='PASS';reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');rows=[]
 for e in reg:
  if not e['has_B']:continue
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;f=np.load(R/e['backfill_features_path'] if e['backfilled'] else R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));t=ev['target'];p0=ev['p0'];ass=ev['assignment'];g0=int(ass[p0.argmax()]);cids=np.flatnonzero(ev['C']);bt=int(cids[t.argmax()]) if len(t) else -1;rows.append({'source':src,'target':tgt,'kind':'late_B' if e['backfilled'] else 'original_B','valid_fraction':float(f['valid_B'].mean()),'supported_groups':int(ev['S'].sum()),'retained_groups':int(ev['C'].sum()),'retained_fraction':float(ev['C'].sum()/max(int(ev['S'].sum()),1)),'conditional_teacher_entropy':float(-(t*np.log(np.maximum(t,1e-300))).sum()),'teacher_top_mass':float(t.max(initial=0)),'teacher_group_equals_zero_group':bool(bt==g0),'zero_group_in_C':bool(ev['C'][g0])})
 out=[]
 for src,tgt in sorted({(x['source'],x['target']) for x in rows}):
  for kind in ['original_B','late_B']:
   rr=[x for x in rows if (x['source'],x['target'],x['kind'])==(src,tgt,kind)]
   if rr:out.append({'source':src,'target':tgt,'kind':kind,'n':len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['valid_fraction','supported_groups','retained_groups','retained_fraction','conditional_teacher_entropy','teacher_top_mass','teacher_group_equals_zero_group','zero_group_in_C']}})
 dump(H/'COVERAGE_METADATA_ONLY.json',{'status':'PRE_TASK_SCORE_DESCRIPTIVE_ONLY','rows':out,'GT_read':False,'no_gate_or_predictor_change':True,'conditioning':'existing R145 thresholds retained, late selection covariate shift not guaranteed calibrated'});print(out,flush=True)
if __name__=='__main__':main()
