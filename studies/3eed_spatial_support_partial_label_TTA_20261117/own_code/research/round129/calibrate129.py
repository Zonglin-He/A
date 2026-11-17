"""One source-only calibration using saved R124 source capacity outputs.

No target inputs, predictions or labels; no new source-model training/inference.
Source-A uncertainty is calibrated with source labels. Source lacks matching B
calibration, so the majority-coverage requirement is a declared fixed convention.
"""
from pathlib import Path
import json,hashlib,numpy as np
from support129 import geometry,glogits_np,soft
R=Path(__file__).resolve().parents[2];H=R/'research/round129';rd=lambda p:json.loads(Path(p).read_text())
def main():
 records=[];counts=[];sourcefiles={};reg=rd(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json')
 for e in reg:
  sid=e['row']['id'];src=e['source'];pp=R/e['files']['prefix']['path'];lp=R/e['files']['label']['path'];sourcefiles[str(pp.relative_to(R))]=e['files']['prefix']['sha256'];sourcefiles[str(lp.relative_to(R))]=e['files']['label']['sha256']
  with np.load(pp) as f:geo=geometry(f['xyz'][0],f['boxes'],np.ones(1024,bool))
  with np.load(lp) as f:y=f['ious']
  qy=np.array([y[g].max() for g in geo['groups']]);good=qy>.25
  if good.any():counts.extend(geo['n_A'][good].tolist())
  f=R/f'crossplatform/round124/formal/{src}/samples_PRIVATE/{sid}.npz'
  with np.load(f) as a:ps=dict(zip(a['arm_names'],a['scores']));p0=ps['zero'].copy();oracle=ps['oracle1'].copy()
  records.append((src,geo,y,qy,p0,oracle))
 minimum=max(2,int(np.floor(np.quantile(counts,.1)))) if counts else 2;gaps=[];bounds=[];coverage=[]
 for src,geo,y,qy,p,oracle in records:
  S=geo['n_A']>=minimum;good=(qy>.25)&S
  if S.sum()<2 or not good.any():coverage.append({'source':src,'source_calibration_eligible':False});continue
  pp=soft(glogits_np(p,geo['groups'])[S]);gap=float(pp.max()-pp[good[S]].max());gaps.append(gap);coverage.append({'source':src,'source_calibration_eligible':True,'gap':gap,'supported_groups':int(S.sum())})
  qS=S[geo['assignment']];a=np.r_[p[~qS],p[qS].sum()];b=np.r_[oracle[~qS],oracle[qS].sum()]
  if y[oracle.argmax()]>y[p.argmax()]+1e-12:bounds.append(float(np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(b,1e-300))))))
 tau=float(np.quantile(gaps,.9)) if gaps else .1;nonzero=[v for v in bounds if v>1e-12];eps=max(1e-6,float(np.quantile(nonzero,.5))) if nonzero else 1e-4
 out={'minimum_support_points':minimum,'minimum_registered_fraction':.5,'tau':tau,'unknown_KL_epsilon':eps,'source_inputs':len(records),'source_eligible_for_tau':len(gaps),'source_task_improving_oracle1_KL_count':len(nonzero),'tau_rule':'pooled source-A 90th percentile gap to any IoU>.25 group under supported group conditional probabilities','epsilon_rule':'pooled median nonzero unknown-collapsed KL of saved R124 source-GT task-improving one-step states; floor1e-6; source-only development','no_new_source_training_or_inference':True,'no_source_B_registration_calibration_available':True,'coverage_majority_is_fixed_convention':True,'target_data_or_GT_loaded':False,'duplicate_rule':'complete-link boxIoU>=.8 AND supportJaccard>=.8, both nonempty; per-group logmeanexp','per_source_or_target_thresholds':False,'source_files':sourcefiles}
 (H/'SOURCE_CALIBRATION.json').write_text(json.dumps(out,indent=2)+'\n');(H/'SOURCE_CALIBRATION_ROWS.json').write_text(json.dumps(coverage,indent=2)+'\n');print({k:v for k,v in out.items() if k!='source_files'})
if __name__=='__main__':main()
