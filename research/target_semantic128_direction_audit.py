"""Source-only algebra audit: common-mean shrink preserves contrast direction."""
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from research.joint_event_v1.run import read,put,sha
A=R/'outputs/target_semantic128_collect_v3';D=R/'outputs/target_semantic128_shrink_v1';P=R/'results/target_semantic128_shrink_v1'
def main():
 start=time.perf_counter();p=read(D/'plan.json');s=read(D/'sealed.json');f=read(D/'frozen_config.json');rows=[];err=0.;rel=0
 for item in p['items']:
  e=s['files'][item['alias']];assert sha(D/e['trace_path'])==e['trace_sha256'];x=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');old=x['steps']['Loop1']['prototypes'];new=x['steps'][f['shrink']]['prototypes']
  for a,b in zip(old,new):
   plus=a['positive_weight'];minus=a['negative_weight'];k=b['kappa'];factor=plus*minus/(plus+minus)*(1/(plus+k)+1/(minus+k));v=a['positive']-a['negative'];w=b['positive']-b['negative'];diff=w-factor*v;err=max(err,float(abs(diff).max()));rel=max(rel,float(np.linalg.norm(diff)/max(np.linalg.norm(w),1e-30)));rows.append(dict(query=item['alias'],dataset=item['dataset'],parity=b['heldout_parity'],contrast_multiplier=factor,contrast_cosine=float(w@v/(np.linalg.norm(w)*np.linalg.norm(v)))if np.linalg.norm(w)*np.linalg.norm(v)>0 else None,max_error=float(abs(diff).max())))
 out=dict(kind='SOURCEONLY exact prototype algebra, no newcandidate orGTscoring',code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),frozen_config_sha256=sha(D/'frozen_config.json'),max_absolute_vector_error=err,max_relative_vector_error=rel,formula='deltaShrink = gamma*deltaOld; gamma=P*N/(P+N)*(1/(P+k)+1/(N+k)), perfitfold P=sumw,N=sum(1-w)',interpretation='With complementaryweights andcommonunconditionalmean, the two centered prototypevectors are collinear. Same scalar kappa rescales their contrast direction. Bernoulli curvature can alter exact L ranking, but firstorder evidence mainly changes strength, not noisy-coordinate direction. This doesnotprove every shrink method useless.',rows=rows,seconds=time.perf_counter()-start,new_model=0,new_GT_read=0,new_predictions=0);put(P/'DIRECTION_AUDIT.json',out);print({k:v for k,v in out.items()if k!='rows'})
if __name__=='__main__':main()
