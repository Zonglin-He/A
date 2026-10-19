"""Independent NumPy fixed-candidate/teacher KL and output-strength diagnostics."""
from pathlib import Path
import os,sys,importlib,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main(kind):
 import numpy as np
 from scripts.audit_stvg_roi_layout_opd32_v1 import sm
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 m=importlib.import_module('scripts.run_stvg_fine_roi_'+kind+'32_v1');p=m.verify();diag=read(m.P/'POST_SEAL_DIAGNOSTICS.json');base=read(m.P/'BASELINE_OPTIMIZATION.json');dd={r['alias']:r for r in diag['rows']};bb={r['alias']:r for r in base['rows']};cnt=collections.Counter();rows=[];maxerr=0.
 def kl(q,pr):return np.where(q>0,q*(np.log(np.maximum(q,1e-300))-np.log(np.maximum(pr,1e-300))),0).sum(-1)
 for r in p['records']:
  o=load(m.D/'predictions'/(r['alias']+'.pt'));old=load(m.CENTER/'predictions'/(r['alias']+'.pt'));fits=dict(Center_live=old['fits']['Fine_live'],**o['fits']);rec={};mask=o['allowed'].numpy();E=o['outputs']['E_MBR']
  for mode,f in fits.items():
   rec[mode]=[]
   for i,t in enumerate(f['rounds']):
    q=t['feedback']['prob'].numpy();pr=t['feedback']['prior'].numpy();valid=t['feedback']['valid'].numpy().copy();valid[~valid.any(1)]=True;d=t['standardized_candidates'].numpy();ps=[sm(np.where(valid,-.5*((d-t[k].numpy()[:,None])**2).sum(-1),-np.inf))for k in ['eta_before','eta_after']];kp=kl(q,pr);kb=kl(q,ps[0]);ka=kl(q,ps[1]);disp=np.linalg.norm((f['path'][i+1]['boxes'].numpy()[:,:2]-f['path'][i]['boxes'].numpy()[:,:2])/E[:,2:],axis=-1);vals=dict(teacher_prior_KL=float(kp[mask].mean()),student_same_teacher_KL_before=float(kb[mask].mean()),student_same_teacher_KL_after=float(ka[mask].mean()),same_teacher_KL_reduction=float((kb-ka)[mask].mean()),normalized_Esize_step_displacement=float(disp[mask].mean()))
    target=bb[r['alias']]['steps'][i]if mode=='Center_live'else dd[r['alias']]['modes'][mode][i]
    for k,v in vals.items():er=abs(v-target[k]);maxerr=max(maxerr,er);assert er<1e-12;cnt['independent_optimization_mean_scalars']+=1
    rec[mode].append(dict(KL_decrease_nodes=int((mask&(ka<kb-1e-12)).sum()),KL_increase_nodes=int((mask&(ka>kb+1e-12)).sum()),KL_neutral_nodes=int((mask&(abs(ka-kb)<=1e-12)).sum()),allowed_nodes=int(mask.sum()),normalized_step_max=float(disp[mask].max())))
   cnt['path_optimization_checks']+=1
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],modes=rec))
 groups={ds:{mode:{str(i+1):{k:paired([r['modes'][mode][i][k]for r in rows if r['dataset']==ds])for k in rows[0]['modes'][mode][i]}for i in range(2)}for mode in rows[0]['modes']}for ds in ['hc2','vidstg']}
 put(m.P/'OPTIMIZATION_DIAGNOSTICS_AUDIT.json',dict(status='pass',count=32,checks=dict(cnt),max_error=maxerr,groups=groups,rows=rows,no_new_GT_or_model=True,same_support_q_and_mask=True));print(kind,'optimization diagnostic audit pass',dict(cnt),flush=True)
if __name__=='__main__':main(sys.argv[1])
