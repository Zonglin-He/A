"""Postseal joint-state response and prelocked decision; no new predictions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_tent_layout_opd32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 p=verify();z=read(P/'RESULTS_PUBLIC.json');rows=[]
 for r in p['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));E=o['outputs']['E_MBR'];stats={}
  for arm,f in o['fits'].items():
   out=f['final'].numpy();delta=(out[:,:2]-E[:,:2])/(.25*E[:,2:]);scope=o['B'].sum(1).numpy()>0;stats[arm]=dict(state_norm=float(f['path'][-1]['state'].norm()),output_center_L2=float(np.linalg.norm(out[:,:2]-E[:,:2])),normalized_first_difference_L2=float(np.linalg.norm(np.diff(delta,axis=0))),normalized_second_difference_L2=float(np.linalg.norm(np.diff(delta,n=2,axis=0))),moved_nodes=int((out[:,:2]!=E[:,:2]).any(1).sum()),steps=[])
   for t in f['rounds']:
    fb=t['feedback'];stats[arm]['steps'].append(dict(prior_ESS=mean((1/fb['prior'].square().sum(-1)).tolist()if arm=='Independent_scoped'else[float(1/fb['prior'].square().sum())]),teacher_ESS=mean((1/fb['prob'].square().sum(-1)).tolist()if arm=='Independent_scoped'else[float(1/fb['prob'].square().sum())]),zero_candidate_prior_mass=mean(fb['prior'][:,0].tolist()if arm=='Independent_scoped'else[float(fb['prior'][0])]),zero_candidate_teacher_mass=mean(fb['prob'][:,0].tolist()if arm=='Independent_scoped'else[float(fb['prob'][0])]),gradient_L2=float(t['gradient'].norm()),sampling_center_L2=float(t['sampling_center'].norm()),candidate_visible_node_fraction=mean(fb['valid'].double().mean(1).tolist())if arm=='Independent_scoped'else float(fb['node_valid'][fb['eligible']].double().mean())if fb['eligible'].any()else None))
  rows.append(dict(alias=a,dataset=r['dataset'],active=o['active'],basis_rank=o['basis_rank'],latent_dimensions=2*int(o['active_axes'].sum()),eligible_nodes=int((o['B'].sum(1)>0).sum()),fits=stats))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,rank_counts={str(k):sum(r['basis_rank']==k for r in rr)for k in range(4)},active=sum(r['active']for r in rr),fits={})
  for arm in ['Joint_fixed','Joint_live','Joint_shuffle','Independent_scoped']:
   groups[ds]['fits'][arm]={key:paired([r['fits'][arm][key]for r in rr])for key in ['state_norm','output_center_L2','normalized_first_difference_L2','normalized_second_difference_L2','moved_nodes']}
   groups[ds]['fits'][arm]['steps']=[{k:paired([r['fits'][arm]['steps'][j][k]for r in rr])for k in rr[0]['fits'][arm]['steps'][j]}for j in range(2)]
 needed=['Joint_live_minus_'+a for a in ['E_MBR','Joint_fixed','Joint_shuffle','Joint_direct','Independent_scoped','Independent_projection','Previous_projection']];gate=all(z['groups'][ds]['paired'][c]['v']['mean']>0 for ds in groups for c in needed)and all(z['groups'][ds]['paired'][needed[0]]['unobserved_actual_s']['mean']>0 for ds in groups)
 put(P/'STATE_RESPONSE_PUBLIC.json',dict(status='complete',groups=groups,rows=rows,native_units_not_pp_for_state_norms=True,no_new_prediction_or_GT_selection=True))
 put(P/'DECISION.json',dict(status='evaluated',passed=gate,rule=p['decision'],all128fits256rounds_actually_ran=True,decision_scope='fixed physical tent joint Layout feedback and two-step categorical output-state update',independent_confirmation=False))
 print('gate',gate,flush=True)
 for ds,g in z['groups'].items():print(ds,{c:{k:round(t[k]['mean'],7)for k in ['v','unobserved_actual_s']}for c,t in g['paired'].items()if c.startswith('Joint_live_')},flush=True)
if __name__=='__main__':main()
