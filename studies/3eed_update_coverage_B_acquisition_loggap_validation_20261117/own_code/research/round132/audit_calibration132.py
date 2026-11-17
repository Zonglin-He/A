"""Recompute source AB log-gap calibration and set retention independently."""
from common132 import *
import numpy as np,collections
def main():
 cal=rd(H132/'SOURCE_LOG_CALIBRATION.json');reg=rd(H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json');gaps=[];rows=[];tau=rd(R132/'research/round129/SOURCE_CALIBRATION.json')['tau']
 for e in reg:
  src=e['entry']['source'];sid=e['entry']['row']['id'];D=B132/'source_calibration_PRIVATE'/src;f=np.load(D/(sid+'.npz'));rec=rd(D/(sid+'.json'));S=f['S'];ass=f['assignment'];groups=[np.flatnonzero(ass==i) for i in range(ass.max()+1)]
  if not rec['B_available'] or S.sum()<2:continue
  with np.load(R132/e['entry']['files']['label']['path']) as y:good=np.array([max(y['ious'][g])>.25 for g in groups])[S]
  if not good.any():continue
  cold=np.zeros(S.sum(),bool);cnew=np.zeros(S.sum(),bool)
  for v in ['A','B']:
   p=np.array([sum(f['p'+v][g])/len(g) for g in groups])[S];p/=sum(p);lg=np.log(np.maximum(p,1e-300));gaps.append(float(max(lg)-max(lg[good])));cold|=max(p)-p<=tau;cnew|=max(lg)-lg<=cal['kappa']
  rows.append({'source':src,'old_retains_good':bool((cold&good).any()),'log_retains_good':bool((cnew&good).any()),'old_fraction':float(cold.mean()),'log_fraction':float(cnew.mean())})
 kappa=float(np.quantile(gaps,.9));assert abs(kappa-cal['kappa'])<1e-12;assert len(gaps)==cal['source_view_gaps']
 out={'status':'PASS_SOURCE_AB_LOG_GAP_QUANTILE_AND_RETENTION','kappa':kappa,'view_gaps':len(gaps),'source_inputs':len(reg),'eligible_inputs':len(rows),'eligible_by_source':dict(collections.Counter(r['source'] for r in rows)),'old_source_AB_good_retention':float(np.mean([r['old_retains_good'] for r in rows])),'log_source_AB_good_retention':float(np.mean([r['log_retains_good'] for r in rows])),'old_mean_retained_fraction':float(np.mean([r['old_fraction'] for r in rows])),'log_mean_retained_fraction':float(np.mean([r['log_fraction'] for r in rows])),'target_labels_read':False,'source_training':False};dump(H132/'SOURCE_CALIBRATION_AUDIT.json',out);print(out)
if __name__=='__main__':main()
