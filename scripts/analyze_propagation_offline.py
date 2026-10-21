"""Read sealed paths/anonymous scores only; no rollouts, models, RGB or new GT."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import sha,read,atomic

def main():
 trial=ROOT/'outputs/propagation_opd_hc64';seal=read(trial/'sealed.json');lock=read(trial/'lock.json')
 public=read(ROOT/'results/propagation_opd_hc64/hc64.json')
 assert seal['queries']==64 and public['proof']['seal_sha256']==sha(trial/'sealed.json')
 thetaE=torch.tensor([0.,.1,1.,.1,.1,0.],dtype=torch.float64)
 rows=[];maxerr=0.;totals={a:dict(changed_queries=0,changed_unobserved_nodes=0,theta_changed_queries=0,visited_nonE=0,visited_active=0)for a in ('OPD','FixedKD','Direct')}
 for index,key in enumerate(lock['ids']):
  entry=seal['files'][key];rp=trial/entry['path'];rr=read(rp.with_suffix('.json'))
  assert sha(rp)==entry['sha256'] and sha(rp.with_suffix('.json'))==entry['receipt_sha256']
  for stem in ('bank','trace','evidence'):assert sha(trial/rr[stem+'_path'])==rr[stem+'_sha256']
  bank=torch.load(trial/rr['bank_path'],map_location='cpu',weights_only=False)
  trace=torch.load(trial/rr['trace_path'],map_location='cpu',weights_only=False)
  obs=bank['observed'];E=bank['E'];query={};theta={}
  for arm in totals:
   tr=trace[arm];changed=(tr['final']!=E).any(-1);theta[arm]=tr['finaltheta'].tolist()
   totals[arm]['changed_queries']+=int(changed.any());totals[arm]['changed_unobserved_nodes']+=int((changed&~obs).sum())
   totals[arm]['theta_changed_queries']+=int(not torch.equal(tr['initialtheta'],tr['finaltheta']))
   for paths in tr['trace']['training_paths']:
    for path in paths:
     active=path['active'];totals[arm]['visited_active']+=int(active.sum())
     totals[arm]['visited_nonE']+=int(((path['actions']!=0)&active).sum())
     valid=path['effective_valid'];features=path['features']
     logits=path['base_logits']+features@thetaE/.1
     logp=torch.log_softmax(logits.masked_fill(~valid,-torch.inf),-1)
     mask=valid
     maxerr=max(maxerr,float((logp[mask]-path['teacher_log_probs'][mask]).abs().max()))
   query[arm]=dict(changed_unobserved_nodes=int((changed&~obs).sum()),final_theta=theta[arm])
  aliases={}
  for a,b in [('OPD','FixedKD'),('OPD','Direct'),('OPD','FixedPropagation')]:
   diff=(trace[a]['final']!=trace[b]['final']).any(-1)
   aliases[a+'-'+b]=dict(any_difference=bool(diff.any()),different_nodes=int(diff.sum()),different_unobserved_nodes=int((diff&~obs).sum()))
  rows.append(dict(query=f'H{index+1:03d}',arms=query,comparisons=aliases))
 comparison={k:dict(queries_with_different_output=sum(r['comparisons'][k]['any_difference']for r in rows),different_nodes=sum(r['comparisons'][k]['different_nodes']for r in rows),different_unobserved_nodes=sum(r['comparisons'][k]['different_unobserved_nodes']for r in rows))for k in rows[0]['comparisons']}
 contributions={}
 for arm in totals:
  d=np.array([100*(r['values'][arm]['v']-r['values']['ERefStrong']['v'])for r in public['queries_anonymous']])
  contributions[arm]=dict(positive_mean_contribution_pp=float(np.maximum(d,0).mean()),negative_mean_contribution_pp=float(np.minimum(d,0).mean()),net_mean_pp=float(d.mean()))
 result=dict(status='sealed_path_audit_complete',queries=64,proof=dict(seal_sha256=sha(trial/'sealed.json'),fingerprint=lock['fingerprint'],GT_reads=0,new_rollouts=0,new_model_calls=0),totals=totals,output_comparisons=comparison,contributions=contributions,teacher_representability=dict(theta_E=thetaE.tolist(),maximum_conditional_logprob_difference_on_saved_states=maxerr,meaning='The enumerated teacher is explicitly representable in the six-weight student on any state. FixedPropagation is its greedy closed-loop path; adaptation is not mathematically required to reproduce this target.',limitations='OPD is hard-history supervised semi-gradient; Direct is full gradient of a soft recurrent relaxation. Their difference cannot identify a universal advantage over direct distillation.'),queries_anonymous=rows)
 assert maxerr<1e-12
 assert sha(trial/'sealed.json')==public['proof']['seal_sha256']
 atomic(ROOT/'results/propagation_opd_hc64/path_audit.json',result)
 print(json.dumps({k:v for k,v in result.items()if k!='queries_anonymous'}))
if __name__=='__main__':main()
