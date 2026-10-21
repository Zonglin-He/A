"""CPU-only matched-initial-state and ROI sampling audit; no decoder/model/GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib,itertools,math,time
from pathlib import Path
import numpy as np,torch
R=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();T=R/'outputs/layout_reward_dev';O=T/'research'
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts/layout_reward'))
from layout_reward import LayoutReward
from stvg.policy import chart,diagonal,noise_for,unchart,log_components,mean_gradient

torch.set_num_threads(1)
SEEDS=[41,42,43,44,45];SCALES=[.05,.25];TAU=.1;LR=.014;EPS=1e-8

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def co(a,b):
 a=a.double().flatten();b=b.double().flatten();den=a.norm()*b.norm();return float(a@b/den)if den>1e-16 else None

def stat(v):
 a=np.asarray([x for x in v if x is not None],float)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))if len(a)else dict(n=0)

def displacement(b1,b0):
 return torch.cat(((b1[:,:2]-b0[:,:2])/b0[:,2:],torch.log(b1[:,2:]/b0[:,2:])),dim=-1)

def group_gradient(mu,actions,D,weights,start,end):
 comp=log_components(mu,actions,D,SCALES);resp=comp.softmax(-1)
 sig=mu.new_tensor(SCALES);inv=(D[:,None,:]*sig[None,:,None]).square().reciprocal()
 each=weights[:,:,None,None]*resp[:,:,:,None]*(mu[:,None,None,:]-actions[:,:,None,:])*inv[:,None,:,:]
 return each[:,start:end].sum((1,2))/len(mu)

start=time.perf_counter();mapping=read(T/'scope_hc2/private_case_mapping.json');records={r['id']:r for r in [json.loads(s)for s in (R/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()]if r['dataset']=='hc2'}
rows=[];alless=[];allentropy=[];allmax=[];allrstd=[];allpairraw=[];allpairwhite=[];allanchorraw=[];grouplarge=[];direct_align=[];verify=dict(path0_box_exact_queries=0,path0_reward_exact_queries=0,seed41_actions_exact_queries=0,seed41_proposal_exact_queries=0,seed41_rewards_max_error=0.,seed41_weights_max_error=0.,seed41_boxes_max_error=0.,Adam_first_step_max_error={'LayoutOPD':0.,'LayoutDirect':0.})
for ii,m in enumerate(mapping):
 key=m['private_id'];assert sha(T/'traces'/f'{key}.pt')==read(T/'predictions'/f'{key}.json')['trace_sha256'];assert sha(T/'student_cache'/f'{key}.pt')==read(T/'student_cache'/f'{key}.json')['sha256'];z=torch.load(T/'traces'/f'{key}.pt',map_location='cpu',weights_only=False);c=torch.load(T/'student_cache'/f'{key}.pt',map_location='cpu',weights_only=False)
 a=z['LayoutOPD'];b=z['LayoutDirect'];base=a['path'][0]['boxes'].double();assert torch.equal(base,b['path'][0]['boxes'].double());verify['path0_box_exact_queries']+=1
 assert torch.equal(a['path'][0]['layout_reward'],b['path'][0]['layout_reward']);verify['path0_reward_exact_queries']+=1
 pos=c['planned_positions'];assert len(pos)==8;obs=np.zeros(len(base),bool);obs[pos]=True;n0=sum(p%2==0 for p in pos);n1=8-n0
 row=dict(id=key,public_id=m['public_id'],positions=pos,offset_anchor_counts=[n0,n1],both_offsets_AH_possible=n0>=2 and n1>=2,initial_state_zero_and_empty_Adam='verified by executed fit code; optimizer buffers were not serialized',same_state_box_exact=True,same_state_reward_exact=True,matched_initial={})
 g0=a['rounds'][0]['gradient'];g1=b['rounds'][0]['gradient'];s0=a['rounds'][0]['state_step'];s1=b['rounds'][0]['state_step'];d0=displacement(a['path'][1]['boxes'].double(),base);d1=displacement(b['path'][1]['boxes'].double(),base)
 row['matched_initial'].update(content_gradient_cosine=co(g0,g1),content_gradient_OPD_to_Direct_norm_ratio=float(g0.double().norm()/g1.double().norm()),Adam_step_cosine=co(s0,s1),Adam_step_OPD_to_Direct_norm_ratio=float(s0.double().norm()/s1.double().norm()),first_output_displacement_cosine=co(d0,d1),first_output_OPD_to_Direct_norm_ratio=float(d0.norm()/d1.norm()),arms={})
 for arm,run in z.items():
  grad=run['rounds'][0]['gradient'];step=run['rounds'][0]['state_step'];expected=-LR*grad/(grad.abs()+EPS);err=float((step-expected).abs().max());verify['Adam_first_step_max_error'][arm]=max(err,verify['Adam_first_step_max_error'][arm]);assert err<1e-6
  dd=displacement(run['path'][1]['boxes'].double(),base);path_sens=[]
  for k,rnd in enumerate(run['rounds']):
   db=displacement(run['path'][k+1]['boxes'].double(),run['path'][k]['boxes'].double());ds=float(rnd['state_step'].double().norm())
   path_sens.append(dict(step=k,state_step_norm=ds,observed_RMS_per_state_norm=float(db[obs].square().mean().sqrt()/ds),unobserved_RMS_per_state_norm=float(db[~obs].square().mean().sqrt()/ds)))
  row['matched_initial']['arms'][arm]=dict(gradient_norm=float(grad.double().norm()),state_delta_norm=float(step.double().norm()),Adam_formula_max_error=err,first_observed_displacement_RMS=float(dd[obs].square().mean().sqrt()),first_unobserved_displacement_RMS=float(dd[~obs].square().mean().sqrt()),first_full_displacement_norm=float(dd.norm()),path_directional_sensitivity=path_sens)
 # Only frozen same-frame ROI reward is evaluated below. No decoder is instantiated.
 reward=LayoutReward(c['teacher']['layout_features'],c['teacher']['layout_mask'],c['teacher']['boxes'],device='cpu');active=reward.active;bs=base[pos];mu_from_CPU_chart=chart(bs);mu=a['rounds'][0]['rollout']['proposal_mean'].double().clone();D=diagonal(bs);row['CPU_vs_saved_GPU_chart_log_max_error']=float((mu-mu_from_CPU_chart).abs().max())
 # Same point-Direct score in box chart, separate from the saved content gradient.
 mp=mu.detach().clone().requires_grad_(True);value=reward(unchart(mp)[:,None]).squeeze(1);dp=torch.autograd.grad(-value[active].mean(),mp)[0].detach()
 seeddata=[];glist=[];gwlist=[]
 for seed in SEEDS:
  with torch.no_grad():
   raw,rng=noise_for(records[key]['key'],seed,0,8,torch.float32);eps=torch.cat((raw[:,:8],raw[:,16:24],raw[:,8:16],raw[:,24:]),1).to(mu)
   sig=mu.new_tensor([SCALES[0]]*16+[SCALES[1]]*16);actions=(mu[:,None]+D[:,None]*sig[None,:,None]*eps).detach();bb=unchart(actions);rr=reward(bb);w=(rr/TAU).softmax(-1)
   gm=mean_gradient(mu[active],actions[active],D[active],w[active],SCALES);full=torch.zeros_like(mu);full[active]=gm;white=full*D
   small=group_gradient(mu[active],actions[active],D[active],w[active],0,16);large=group_gradient(mu[active],actions[active],D[active],w[active],16,32);assert torch.allclose(small+large,gm,rtol=1e-12,atol=1e-12)
   ess=1/w.square().sum(-1);ent=-(w*w.clamp_min(1e-300).log()).sum(-1)
   glist.append(full);gwlist.append(white);alless.extend(ess[active].tolist());allentropy.extend(ent[active].tolist());allmax.extend(w[active].max(-1).values.tolist());allrstd.extend(rr[active].std(-1).tolist());grouplarge.extend(w[active,16:].sum(-1).tolist())
   norm2=gm.square().sum();projection=float((large*gm).sum()/norm2)if norm2>1e-20 else None
   sd=dict(seed=seed,rng_seed=rng,box_gradient=full.tolist(),D_scaled_box_gradient=white.tolist(),ESS_mean=float(ess[active].mean()),ESS_std_across_anchors=float(ess[active].std(unbiased=False)),max_weight_mean=float(w[active].max(-1).values.mean()),reward_std_mean=float(rr[active].std(-1).mean()),large_proposal_probability=float(w[active,16:].sum(-1).mean()),small_gradient_norm=float(small.norm()),large_gradient_norm=float(large.norm()),large_gradient_projection_on_total=projection,small_large_gradient_cosine=co(small,large),point_Direct_box_gradient_cosine=co(full,dp),D_scaled_point_Direct_box_gradient_cosine=co(white,dp*D))
   if seed==41:
    saved=a['rounds'][0]['rollout'];sd['saved_actions_bit_exact']=torch.equal(actions,saved['actions']);verify['seed41_actions_exact_queries']+=int(sd['saved_actions_bit_exact']);assert sd['saved_actions_bit_exact']
    sd['saved_proposal_bit_exact']=torch.equal(mu,saved['proposal_mean']);verify['seed41_proposal_exact_queries']+=int(sd['saved_proposal_bit_exact']);assert sd['saved_proposal_bit_exact']
    for name,new,old in [('boxes',bb,saved['boxes']),('rewards',rr,saved['rewards']),('weights',w,saved['weights'])]:
     error=float((new-old).abs().max());sd[f'saved_{name}_max_error']=error;verify[f'seed41_{name}_max_error']=max(error,verify[f'seed41_{name}_max_error'])
    assert sd['saved_boxes_max_error']<1e-12 and sd['saved_rewards_max_error']<5e-6 and sd['saved_weights_max_error']<1e-5
    direct_align.append(sd['point_Direct_box_gradient_cosine'])
   seeddata.append(sd)
 pairs=[]
 for i,j in itertools.combinations(range(5),2):
  cr=co(glist[i],glist[j]);cw=co(gwlist[i],gwlist[j]);anchor=[co(glist[i][k],glist[j][k])for k in range(8)if active[k]];allpairraw.append(cr);allpairwhite.append(cw);allanchorraw.extend(anchor);pairs.append(dict(seed0=SEEDS[i],seed1=SEEDS[j],gradient_cosine=cr,D_scaled_gradient_cosine=cw,anchor_gradient_cosines=anchor))
 gs=torch.stack(glist);gmean=gs.mean(0);within=float((gs-gmean).square().sum(1).sum(1).mean().sqrt());den=float(gmean.norm());row['sampling']=dict(seeds=seeddata,seed_pairs=pairs,gradient_deviation_RMS_over_mean_norm=within/den if den>1e-16 else None,relative_seed_noise_is_output_chart_not_content=True)
 rows.append(row)
 if (ii+1)%16==0:print(json.dumps(dict(done=ii+1,total=128,seconds=time.perf_counter()-start)),flush=True)

summary=dict(queries=128,matched_initial={k:stat([r['matched_initial'][k]for r in rows])for k in ['content_gradient_cosine','content_gradient_OPD_to_Direct_norm_ratio','Adam_step_cosine','Adam_step_OPD_to_Direct_norm_ratio','first_output_displacement_cosine','first_output_OPD_to_Direct_norm_ratio']},verification=verify,sampling=dict(seeds=SEEDS,point='initial ERef fixed state, step0 only',box_gradient_seed_pair_cosine=stat(allpairraw),D_scaled_box_gradient_seed_pair_cosine=stat(allpairwhite),anchor_box_gradient_seed_pair_cosine=stat(allanchorraw),seed_pairs_negative_cosine_fraction=float(np.mean([v<0 for v in allpairraw if v is not None])),anchor_seed_pairs_negative_cosine_fraction=float(np.mean([v<0 for v in allanchorraw if v is not None])),ESS=stat(alless),weight_entropy_nats=stat(allentropy),maximum_weight=stat(allmax),candidate_reward_std=stat(allrstd),large_proposal_probability=stat(grouplarge),seed41_point_Direct_box_gradient_cosine=stat(direct_align),gradient_deviation_RMS_over_mean_norm=stat([r['sampling']['gradient_deviation_RMS_over_mean_norm']for r in rows])),offset_support=dict(count_pairs={str(k):sum(tuple(r['offset_anchor_counts'])==k for r in rows)for k in sorted({tuple(r['offset_anchor_counts'])for r in rows})},one_or_more_empty_offset=sum(min(r['offset_anchor_counts'])==0 for r in rows),one_or_more_single_anchor_offset=sum(1 in r['offset_anchor_counts']for r in rows),A_H_each_offset_feasible=sum(r['both_offsets_AH_possible']for r in rows)),CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start)
for arm in ['LayoutOPD','LayoutDirect']:
 summary['matched_initial'][arm]={k:stat([r['matched_initial']['arms'][arm][k]for r in rows])for k in ['gradient_norm','state_delta_norm','first_observed_displacement_RMS','first_unobserved_displacement_RMS','first_full_displacement_norm']}
 pp=[p for r in rows for p in r['matched_initial']['arms'][arm]['path_directional_sensitivity']]
 summary.setdefault('path_sensitivity',{})[arm]={k:stat([p[k]for p in pp])for k in ['observed_RMS_per_state_norm','unobserved_RMS_per_state_norm']}
summary['verification'].update(seed41_CPU_vs_saved_GPU_chart_log_max_error=max(r['CPU_vs_saved_GPU_chart_log_max_error']for r in rows),trace_and_student_cache_sha256_queries=128)
summary['sampling']['scale_contributions']={k:stat([s[k]for r in rows for s in r['sampling']['seeds']])for k in ['small_gradient_norm','large_gradient_norm','large_gradient_projection_on_total','small_large_gradient_cosine','D_scaled_point_Direct_box_gradient_cosine']}
summary['sampling']['anchor_zero_gradient_pair_cosines_undefined']=sum(c is None for r in rows for p in r['sampling']['seed_pairs']for c in p['anchor_gradient_cosines'])
summary['matched_initial']['negative_content_gradient_cosine_queries']=sum(r['matched_initial']['content_gradient_cosine']<0 for r in rows)
summary['matched_initial']['negative_Adam_step_cosine_queries']=sum(r['matched_initial']['Adam_step_cosine']<0 for r in rows)
summary['matched_initial']['negative_first_output_displacement_cosine_queries']=sum(r['matched_initial']['first_output_displacement_cosine']<0 for r in rows)
summary['offset_support']['AH_exception_public_ids']=[r['public_id']for r in rows if not r['both_offsets_AH_possible']]
summary['limits']=['Only k=0 compares identical E/content state and freshly instantiated Adam; later arms visit different states and cannot diagnose optimizer preconditioning under matched state.','CPU seed41 actions must be bit exact; CPU/GPU bilinear sampling and FP32 reduction may differ within explicitly recorded tolerances.','Five-seed repetition evaluates frozen ROI rewards and box-chart score gradients only, not decoder rollouts, content gradients or new task predictions.','Path sensitivity is finite displacement along actually executed Adam directions, not a complete Jacobian, rank or controllability result.']
assert not torch.cuda.is_initialized();(O/'matched_state_sampling.json').write_text(json.dumps(dict(summary=summary,queries=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))
