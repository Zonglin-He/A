"""CPU arithmetic/Adam/state/official metric audit; no GPU replay claim."""
import os,sys,time,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_sparse_state_route32_v1 import D,P,read,put,load,sha,verify

def loss_numpy(boxes,positions,targets):
 import numpy as np
 if not positions:return 0.
 b=np.asarray(boxes)[positions];t=np.asarray(targets);lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;tl=t[:,:2]-t[:,2:]/2;th=t[:,:2]+t[:,2:]/2;i=np.maximum(0,np.minimum(hi,th)-np.maximum(lo,tl)).prod(-1);u=b[:,2:].prod(-1)+t[:,2:].prod(-1)-i;c=np.maximum(0,np.maximum(hi,th)-np.minimum(lo,tl)).prod(-1);return float((5*abs(b-t).sum(1)+2*(1-i/np.maximum(u,1e-30)+(c-u)/np.maximum(c,1e-30))).mean())

def independent_paired(values):
 import numpy as np
 v=np.asarray([x for x in values if x is not None]);return (None,None,0)if not len(v)else(float(v.mean()),np.quantile(np.random.default_rng(20261018).choice(v,(10000,len(v))).mean(1),[.025,.975]),len(v))

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 torch.set_num_threads(4);start=time.perf_counter();p=verify(full=True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_actions_states_before_GT'];result=read(P/'RESULTS_PUBLIC.json');assert len(result['rows'])==32;checks=collections.Counter();errors=collections.Counter();details=[]
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 def near(a,b,k,tol=1e-10):
  e=float(np.max(abs(np.asarray(a)-np.asarray(b))));errors[k]=max(errors[k],e);assert e<tol,(k,e)
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'));out=next(x for x in result['rows']if x['alias']==r['alias']);S=z['Source'].double().numpy();E=z['E'].numpy();targets=z['targets'];positions=z['positions'];fix=z['fits']['State_fixed'];live=z['fits']['State_live'];assert z['zero_Source_E_exact']and np.array_equal(fix['path'][1]['state'],live['path'][1]['state'])and np.array_equal(fix['rounds'][0]['gradient'],live['rounds'][0]['gradient']);checks['first_step_fixed_live_exact']+=1
  def retained(native):
   n=np.asarray(native,dtype=float);return np.concatenate([E[:,:2]+n[:,:2]-S[:,:2],E[:,2:]*(n[:,2:]/S[:,2:])],1)
  for arm,f in z['fits'].items():
   m=np.zeros(256);v=np.zeros(256);state=np.zeros(256);optstep=0
   for j,t in enumerate(f['rounds']):
    before=t['before'];assert np.array_equal(before,state);near(t['loss_before'],loss_numpy(f['path'][j]['boxes'],positions,targets),'loss_before');near(t['loss_after'],loss_numpy(t['post_boxes'],positions,targets),'loss_after');near(t['post_boxes'],retained(t['post_native']),'E_retaining',2e-12);g=np.asarray(t['gradient']);oldstate=state.copy()
    if arm!='Direct_anchor'and positions:
     proposal=np.zeros(256)if arm=='State_fixed'else before;assert np.array_equal(proposal,t['proposal']);assert np.array_equal(t['actions'],proposal[None]+z['noise'][j]);assert np.array_equal(z['noise'][j,:8],-z['noise'][j,8:]);losses=[]
     for rec in t['actual_candidates']:
      near(rec['boxes'],retained(rec['native']),'candidate_E_mapping',2e-12);losses.append(loss_numpy(rec['boxes'],positions,targets));checks['actual_sampled_trajectories_audited']+=1
     near(t['losses'],losses,'candidate_losses');valid=t['valid'];lp=-.5*np.square(t['actions']-before).sum(1)+.5*np.square(t['actions']-proposal).sum(1);lp[~valid]=-np.inf
     def sm(a):
      b=np.exp(a-np.max(a));return b/b.sum()
     base=sm(lp);informative=bool(np.ptp(np.asarray(losses)[valid])>1e-12);teacher=sm(lp-np.asarray(losses))if informative else base.copy();near(t['weights']['base'],base,'RNbase');near(t['weights']['teacher'],teacher,'teacher_weights');eg=((base-teacher)[:,None]*t['actions']).sum(0)if informative else np.zeros(256);near(g,eg,'OPD_gradient');checks['actual_RN_updates']+=1
    elif arm=='Direct_anchor'and positions:checks['recorded_decoder_direct_gradients']+=1
    else:assert not t['updated']and np.array_equal(g,np.zeros(256))
    if t['updated']:
     # FP64 independent Adam reference; worker applies FP32 CUDA Adam.
     gg=g.astype(np.float32).astype(float);optstep+=1;m=.9*m+.1*gg;v=.999*v+.001*gg*gg;state=(state-p['Adam'][r['dataset']]*(m/(1-.9**optstep))/(np.sqrt(v/(1-.999**optstep))+1e-8)).astype(np.float32).astype(float);near(state,t['after'],'Adam_state',1e-6);state=t['after'].copy();adam=next(iter(t['Adam']['state'].values()));assert int(adam['step'])==optstep;near(adam['exp_avg'],m,'Adam_moment',1e-6);near(adam['exp_avg_sq'],v,'Adam_second_moment',1e-6)
    else:assert np.array_equal(t['after'],before);state=before.copy();checks['exact_noop_steps']+=1
    assert np.array_equal(f['path'][j+1]['state'],state)and np.array_equal(f['path'][j+1]['boxes'],t['post_boxes']);checks['scheduled_steps']+=1
   assert np.array_equal(f['final'],z['outputs'][arm]);checks['mean_readouts']+=1
  if z['gradient_probe']is not None:assert z['gradient_probe']['absolute_error']<z['gradient_probe']['tolerance'];checks['GPU_FD_probes_receipts']+=1
  assert z['compute']['prefix_native_checks']>=2 and z['compute']['actual_PosDecoder_calls']==2*z['compute']['actual_full_trajectory_decoder_calls'];checks['prefix_invariance_and_frozen_hash_receipts']+=1
  ds=r['dataset'];meta=plans[ds][r['query_ordinal']];truth,span=gt[ds][0][r['query_ordinal']],gt[ds][1][r['query_ordinal']];ids=np.array(z['frame_ids']);known=np.array([int(fid)in truth for fid in ids]);observed=np.isin(np.arange(len(ids)),z['planned_positions']);assert int((known&~observed).sum())==out['known_unobserved_actual'];checks['original_parent_denominators']+=1
  for arm,b in z['outputs'].items():
   actual=official(b,meta,truth,span,z['interval'],ds);tube=DenseTube(b,meta,truth,span,clip=ds=='hc2');si=np.array([iou_gt(b[i],truth[int(fid)],meta,ds)if known[i]else np.nan for i,fid in enumerate(ids)]);interp=~np.isin(tube.fids,ids);actual.update(unobserved_actual_s=float(si[known&~observed].mean())if(known&~observed).any()else None,observed_actual_s=float(si[known&observed].mean())if(known&observed).any()else None,benchmark_interpolated_s=float(tube.iou[interp].mean())if interp.any()else None)
   for k,value in actual.items():
    y=out['values'][arm][k]
    if value is None:assert y is None
    else:near(value,y,'official_metric',1e-12)
    checks['metric_values']+=1
  details.append(dict(alias=r['alias'],dataset=ds,actual_decoder_calls=z['compute']['actual_full_trajectory_decoder_calls'],observed_pseudo_loss_change={a:f['rounds'][-1]['loss_after']-f['rounds'][0]['loss_before']for a,f in z['fits'].items()},prefix_unchanged=True,source_E_zero_exact=True))
 for ds,g in result['groups'].items():
  rr=[r for r in result['rows']if r['dataset']==ds];assert len(rr)==16
  for pair,eps in g['paired'].items():
   a,b=pair.split('_minus_')
   for e,s in eps.items():
    mean,ci,n=independent_paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr]);assert n==s['parents']
    if mean is None:assert s['mean']is None
    else:near(mean,s['mean'],'parent_mean');near(ci,s['CI95'],'bootstrap_CI')
    checks['parent_bootstrap_contrasts']+=1
 verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),max_absolute_errors=dict(errors),all32_before_GT_barrier=True,actual_GPU_decoder_not_independently_replayed=True,direct_gradients_engineering_supported_by_fixed_FD_probes_and_site_invariance=True,old_baseline_and_formal64_and_registries_unchanged=True,unknown_and_original_denominators_preserved=True,CPU_seconds=time.perf_counter()-start));put(P/'ACTUAL_STATE_AUDIT_PUBLIC.json',dict(rows=details,count=32));print('root independent audit passed',dict(checks),dict(errors),flush=True)
if __name__=='__main__':main()
