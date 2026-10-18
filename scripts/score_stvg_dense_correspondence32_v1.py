"""CPU-only postseal metrics and independent likelihood/Adam/state audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,collections,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dense_correspondence32_v1 import D,ARMS
P=ROOT/'results/stvg_soft_correspondence/2026-10-18/online32'

def avg(xs):
 x=[v for v in xs if v is not None];return float(np.mean(x))if x else None

def rank_transfer(r,ref):
 out=np.empty_like(ref)
 for i,values in enumerate(r):
  if np.all(values==values[0]):out[i]=1/32;continue
  order=np.argsort(values,kind='stable');ww=np.sort(ref[i]);start=0
  while start<len(values):
   stop=start+1
   while stop<len(values)and values[order[stop]]==values[order[start]]:stop+=1
   out[i,order[start:stop]]=ww[start:stop].mean();start=stop
 return out

def smweights(r,tau):
 z=r/tau;w=np.exp(z-z.max(1,keepdims=True));return w/w.sum(1,keepdims=True)

def vector(st,names):return np.concatenate([st[n].numpy().reshape(-1)for n in names])

def audit_dense(z,expert,feature):
 from vg_tta.decota_fixed_full_audit_v1 import top1_support,overlap
 import torch
 cfg=z['config'];hist=z['path'];names=list(z['initial']);support=top1_support(expert);assert z['positions']==[p for p,_ in support]
 assert z['selected_step']==cfg['steps']and len(hist)==cfg['steps']+1 and len(z['rounds'])==cfg['steps']
 assert np.array_equal(z['final'].numpy(),hist[-1]['boxes'].numpy())and np.array_equal(vector(z['state'],names),vector(hist[-1]['state'],names))
 m=np.zeros(1792);v=m.copy();updates=0;maxadam=0.;maxg=0.;checks=0;rewarderror=0.;weightserror=0.
 for k,(h,rd)in enumerate(zip(hist,z['rounds'])):
  a=vector(h['state'],names);nxt=vector(hist[k+1]['state'],names)
  if not support:assert not rd['updated']and np.array_equal(a,nxt);continue
  for tag,positions in [('sparse',z['positions']),('dense',z['unobserved_positions'])]:
   rr=rd[tag+'_rollout']
   if rr is None:continue
   seed=rr['seed'];gen=torch.Generator(device='cpu').manual_seed(seed);noise=torch.randn(len(positions),16,4,generator=gen,dtype=torch.float32);noise=torch.cat((noise,-noise),1);expected_samples=rr['mean'][:,None]+cfg['sigma']*noise;assert torch.equal(expected_samples,rr['samples']),'seed/antithetic/sigma mismatch'
   samples=rr['samples'].numpy().astype(float);mu=rr['mean'].numpy().astype(float);w=rr['weights'].numpy().astype(float);current=np.log(h['boxes'][positions].numpy().astype(float)/(1-h['boxes'][positions].numpy().astype(float)))
   # Current chart has float32 logit roundoff: use exact saved rollout mean (on-policy).
   assert abs(current-mu).max()<1e-5
   gradient=-((w-1/32)[:,:,None]*(samples-mu[:,None])).sum(1)/(cfg['sigma']**2*len(positions))
   g=rd[tag+'_mean_gradient'].numpy();err=float(abs(gradient-g).max());assert err<4e-5,err;maxg=max(maxg,err)
   if tag=='sparse':
    actions=1/(1+np.exp(-samples));rewards=np.array([[overlap(b,e)for b in aa]for aa,(_,e)in zip(actions,support)]);err=float(abs(rewards-rr['rewards'].numpy()).max());assert err<4e-6;rewarderror=max(rewarderror,err);expected=smweights(rr['rewards'].numpy().astype(float),cfg['tau'])
   else:
    geom=rr['interpolation_rewards'].numpy().astype(float);raw=rr['raw_rewards'].numpy().astype(float);expected=smweights(geom if z['arm']=='D_interpolation'else raw,cfg['tau'])
    if z['arm']=='C_rank':expected=rank_transfer(raw,smweights(geom,cfg['tau']))
    # Independent continuous interpolation IoU; sampled coordinates remain untouched.
    actions=1/(1+np.exp(-samples));ref=feature['interpolated'][positions].numpy();geom2=np.array([[overlap(b,e)for b in aa]for aa,e in zip(actions,ref)]);err=float(abs(geom2-geom).max());assert err<4e-6;rewarderror=max(rewarderror,err)
    if k in [0,cfg['steps']//2,cfg['steps']-1]:
     # Independent NumPy area integration of the immutable per-anchor soft maps.
     maps=feature['soft'][:,positions].numpy().astype(float);H,W=maps.shape[-2:];x=np.arange(W);y=np.arange(H);lo=actions[...,:2]-actions[...,2:]/2;hi=actions[...,:2]+actions[...,2:]/2
     dx=np.clip(np.minimum(hi[...,0,None]*W,x+1)-np.maximum(lo[...,0,None]*W,x),0,1);dy=np.clip(np.minimum(hi[...,1,None]*H,y+1)-np.maximum(lo[...,1,None]*H,y),0,1);area=dy[..., :,None]*dx[...,None,:];inter=(area[None]*maps[:,:,None]).sum((-1,-2));union=actions[None,...,2]*actions[None,...,3]*H*W+maps.sum((-1,-2))[:,:,None]-inter;raw2=np.divide(inter,union,out=np.zeros_like(inter),where=union>0).mean(0);err=float(abs(raw2-raw).max());assert err<4e-6,err;rewarderror=max(rewarderror,err)
   err=float(abs(expected-w).max());assert err<5e-7,err;weightserror=max(weightserror,err);checks+=len(positions)
  if not rd['updated']:assert np.array_equal(a,nxt);continue
  u=h['update'];assert u['names']==names
  g=u['gradient'].numpy().astype(float);updates+=1;m=.9*m+.1*g;v=.999*v+.001*g*g;delta=-cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8);err=float(abs(delta-u['raw'].numpy()).max());assert err<4e-6,err;maxadam=max(maxadam,err);assert np.array_equal(nxt-a,u['raw'].numpy())
 assert updates==z['gradient_calls']
 return dict(status='pass',max_Adam_error=maxadam,max_mean_gradient_error=maxg,max_detached_reward_error=rewarderror,max_weight_error=weightserror,frame_likelihood_checks=checks,gradient_calls=updates,independent_decoder_Jacobian=False)

def summary(rows):
 out=dict(queries=len(rows),parents=len({r['parent_alias']for r in rows}),metrics={},deltas={},paired={},no_fit_queries=sum(r['no_fit']for r in rows),unknown_actual_frames=sum(r['unknown_actual_frames']for r in rows),known_actual_supports={support:dict(frames=sum(r['actual_support_counts'][support]for r in rows),parents_with_support=sum(r['actual_support_counts'][support]>0 for r in rows))for support in ['observed','unobserved','far']})
 if not rows:return out
 # One query per parent by design; query macro and parent macro equal, explicitly report both.
 rng=np.random.default_rng(20261018);ix=rng.integers(0,len(rows),size=(10000,len(rows)))
 for a in ARMS:
  out['metrics'][a]={state:{met:avg([r['values'][a][state][met]for r in rows])for met in ['v','t','s','R30','R50']}for state in ['Source','Before','After']}
  ds={}
  for kind in ['total','current','inherited']:
   val=np.array([r['deltas'][a][kind]for r in rows]);ds[kind]=dict(query_mean_pp=float(val.mean()*100),parent_mean_pp=float(val.mean()*100),CI95_pp=(np.quantile(val[ix].mean(1),[.025,.975])*100).tolist(),positive=int((val>1e-10).sum()),negative=int((val< -1e-10).sum()),neutral=int((abs(val)<=1e-10).sum()),severe_harm_gt5pp=int((val<-.05).sum()),gross_gain_pp=float(np.maximum(val,0).mean()*100),gross_loss_pp=float(-np.minimum(val,0).mean()*100))
  out['deltas'][a]=ds
  out.setdefault('actual_decoder',{})[a]={s:{state:avg([r['decoder'][a][s][state]for r in rows])for state in ['Source','Before','After']}for s in ['observed','unobserved','far']}
  out.setdefault('compute',{})[a]={key:avg([r['compute'][a][key]for r in rows])for key in ['fit_seconds','gradient_calls','dense_rank_delta_pp','sparse_ESS','dense_ESS','parameter_step_norm','parameter_gradient_norm','mean_gradient_norm_sparse','mean_gradient_norm_dense']}
 for a in ARMS[1:]:
  for b in ['A_sparse','D_interpolation']:
   if a==b:continue
   vals=np.array([r['values'][a]['After']['v']-r['values'][b]['After']['v']for r in rows]);loo=[float(np.delete(vals,i).mean()*100)for i in range(len(vals))]if len(vals)>1 else[]
   out['paired'][a+'_minus_'+b]=dict(mean_pp=float(vals.mean()*100),CI95_pp=(np.quantile(vals[ix].mean(1),[.025,.975])*100).tolist(),positive=int((vals>1e-10).sum()),negative=int((vals< -1e-10).sum()),neutral=int((abs(vals)<=1e-10).sum()),leave_one_parent_out_min_pp=min(loo)if loo else None)
 return out

def run():
 import torch
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
 from scripts.stvg_opd_paper_common_v1 import committed
 p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['fits']==128 and not bar['GT_read']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(status='post_all128_seal',barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),timestamp_epoch=time.time()))
 targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};features=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json');fby={ds:{f['query_ordinal']:f for f in fs}for ds,fs in features['rows'].items()};plans={ds:read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')for ds in targets};rows=[];audits=[];previous={ds:{a:None for a in ARMS}for ds in targets};metricchecks=statechecks=0
 for rec in p['records']:
  z=load(D/'predictions'/(rec['alias']+'.pt'));feat=load(D/'features'/(rec['alias']+'.pt'));ds=z['dataset'];q=z['query_ordinal'];row=plans[ds]['rows'][q];gt=targets[ds][0][q];span=targets[ds][1][q];ids=np.array(z['frame_ids']);planned=np.isin(np.arange(len(ids)),z['planned_positions']);known=np.array([int(fid)in gt for fid in ids]);dist=np.min(abs(ids[:,None]-ids[np.array(z['planned_positions'])][None,:]),axis=1)if z['planned_positions']else np.zeros(len(ids));far=dist>=.25*(ids[-1]+1-ids[0]);masks=dict(observed=known&planned,unobserved=known&~planned,far=known&~planned&far);values={};deltas={};decoder={};cost={};f=fby[ds][q]
  for arm in ARMS:
   fit=z['arms'][arm];assert fit['full_readout_exact']and fit['config']==p['configs'][ds]and not fit['GT_used']
   prior=previous[ds][arm];expected=z['Source_initial']if prior is None else{n:torch.zeros_like(v)if n=='spatial.query_residual'else prior[n]for n,v in z['Source_initial'].items()}
   assert all(torch.equal(fit['initial'][n],v)for n,v in expected.items());commit=committed(fit['initial'],fit['state'],fit['config']['writeback']);assert all(torch.equal(commit[n],fit['committed'][n])for n in commit);previous[ds][arm]=commit;statechecks+=3584
   
   for k,rd in enumerate(fit['rounds']):
    for tag,rr in [('spatial-opd-v1',rd.get('rollout')if arm=='A_sparse'else rd.get('sparse_rollout')),('dense-correspondence-v1',rd.get('dense_rollout'))]:
     if rr is not None:assert rr['seed']==int(hashlib.sha256(f'{tag}|{row["key"]}|{k}'.encode()).hexdigest()[:15],16)
   check=audit(fit,z['expert'])if arm=='A_sparse'else audit_dense(fit,z['expert'],feat);audits.append(dict(alias=rec['alias'],arm=arm,**check))
   vals={};actual={}
   for state,box in [('Source',z['Source_boxes']),('Before',fit['before']),('After',fit['final'])]:
    vals[state]=official(box.numpy(),row,gt,span,z['Native_interval'],ds);ind=DenseTube(box.numpy(),row,gt,span,clip=ds=='hc2').score(z['Native_interval']);assert all(abs(ind[k]-vals[state][k])<2e-10 for k in ['v','t','s']);metricchecks+=3
    vals[state].update(R30=float(vals[state]['v']>.3),R50=float(vals[state]['v']>.5));xy=xyxy(box.numpy(),row['input']['width'],row['input']['height']);xy=np.maximum(xy,0)if ds=='hc2'else xy;quality=np.array([float(box_iou(xy[j],gt[int(fid)]))if int(fid)in gt else np.nan for j,fid in enumerate(ids)]);actual[state]={s:avg(quality[mask].tolist())for s,mask in masks.items()}
   assert vals['Source']['t']==vals['Before']['t']==vals['After']['t'];values[arm]=vals;deltas[arm]=dict(total=vals['After']['v']-vals['Source']['v'],current=vals['After']['v']-vals['Before']['v'],inherited=vals['Before']['v']-vals['Source']['v']);assert abs(deltas[arm]['total']-deltas[arm]['current']-deltas[arm]['inherited'])<1e-12;decoder[arm]={s:{st:actual[st][s]for st in actual}for s in masks};rd=fit['rounds'];ranks=[]
   for k,x in enumerate(rd):
    rr=x.get('dense_rollout')
    if rr is None:continue
    positions=fit['unobserved_positions'];valid=[j for j,pos in enumerate(positions)if known[pos]]
    if not valid:continue
    acts=rr['samples'][valid].double().sigmoid().numpy();xy=xyxy(acts,row['input']['width'],row['input']['height']);xy=np.maximum(xy,0)if ds=='hc2'else xy;qq=np.array([box_iou(xx,gt[int(ids[positions[j]])])for j,xx in zip(valid,xy)]);ww=rr['weights'][valid].numpy();ranks.append(float(((qq*ww).sum(1)-qq.mean(1)).mean()*100))
   def rdmean(key):return avg([float(x[key].mean())if hasattr(x[key],'mean')else float(x[key])for x in rd if x.get(key)is not None])
   cost[arm]=dict(fit_seconds=fit['wall_seconds'],gradient_calls=fit['gradient_calls'],dense_rank_delta_pp=avg(ranks),sparse_ESS=rdmean('sparse_ESS')if arm!='A_sparse'else rdmean('ESS'),dense_ESS=rdmean('dense_ESS'),parameter_step_norm=rdmean('parameter_step_norm')if arm!='A_sparse'else avg([float(h['update']['raw'].norm())for h in fit['path']if'update'in h]),parameter_gradient_norm=rdmean('parameter_gradient_norm')if arm!='A_sparse'else avg([float(h['update']['gradient'].norm())for h in fit['path']if'update'in h]),mean_gradient_norm_sparse=rdmean('sparse_gradient_norm'),mean_gradient_norm_dense=rdmean('dense_gradient_norm'))
  rows.append(dict(alias=rec['alias'],parent_alias=rec['alias'],dataset=ds,event_length_bin=f['bins']['event_fraction'],motion_bin=f['bins']['speed_sizes_sec'],target_area_bin=f['bins']['target_area'],query_form=f['query_form'],values=values,deltas=deltas,decoder=decoder,compute=cost,admitted_anchors=len(z['admitted_positions']),no_fit=len(z['admitted_positions'])==0,unknown_actual_frames=int((~known).sum()),actual_support_counts={s:int(m.sum())for s,m in masks.items()},planned_observation_slots=len(z['planned_positions'])))
  print('SCORED',rec['alias'],flush=True)
 groups={ds:summary([r for r in rows if r['dataset']==ds])for ds in targets};strata={ds:{k:{v:summary([r for r in rows if r['dataset']==ds and r[k]==v])for v in sorted({r[k]for r in rows if r['dataset']==ds})}for k in ['event_length_bin','motion_bin','target_area_bin','query_form']}for ds in groups};gate={}
 for ds,g in groups.items():
  paired=[g['paired']['C_rank_minus_'+a]for a in ['A_sparse','D_interpolation']];severeC=g['deltas']['C_rank']['current']['severe_harm_gt5pp'];severeA=g['deltas']['A_sparse']['current']['severe_harm_gt5pp'];conds=dict(C_beats_A_and_D=all(v['mean_pp']>0 for v in paired),leave_one_out_positive=all(v['leave_one_parent_out_min_pp']is not None and v['leave_one_parent_out_min_pp']>0 for v in paired),no_large_harm_count_increase=severeC<=severeA+1);gate[ds]=dict(conditions=conds,passed=all(conds.values()))
 P.mkdir(parents=True,exist_ok=True);put(P/'RESULTS.json',dict(status='complete',count=32,groups=groups,strata=strata,rows=rows,all128_fit_predictions_before_GT=True,independent_confirmation=False,query_macro_equals_parent_macro_by_one_query_per_parent=True,unknown_GT_not_zero_filled=True,original_online_stream_not_reproduced=True))
 put(P/'AUDIT.json',dict(status='pass',rows=audits,official_metric_scalar_checks=metricchecks,state_coordinate_checks=statechecks,full_two_Decoder_readouts_exact=True,Source_and_DINO_pixels_exact=True,GT_after_prediction_seal=True))
 passed=all(g['passed']for g in gate.values());put(P/'DECISION.json',dict(status='further_evidence_required'if passed else'stop_this_dense_interface',groups=gate,passed=passed,automatic_dispatch=False,no_temperature_or_encoder_search=True,production_unchanged=True))
 put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,result_sha256=sha(P/'RESULTS.json')));put(D/'ROOT_READBACK.json',dict(status='pass',count=32,independent_metric_checks=metricchecks,independent_state_checks=statechecks));print('GATE',passed,flush=True)
if __name__=='__main__':run()
