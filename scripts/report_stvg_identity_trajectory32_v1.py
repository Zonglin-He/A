"""Independent postseal audit, full signed scalar publication and fixed pixel panels."""
from pathlib import Path
import argparse,collections,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_identity_trajectory32_v1 import BASE,OLD,ARMS
PUB=ROOT/'results/stvg_identity_trajectory/2026-10-18/matched32'
def score():
 import numpy as np,torch
 from scipy.special import logsumexp
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit as old_audit
 from vg_tta.stvg_identity_trajectory_opd_v1 import covariance
 p=read(BASE/'PLAN_LOCK.json');bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==32
 zs=[load(ROOT/f['path'])for f in bar['files']];assert len(zs)==32
 targets={d:truths(d,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==d]))for d in ['hc2','vidstg']}
 rows=[];hypmeta=[];mathchecks=checks=0;maxdensity=0.;cost=collections.defaultdict(list)
 for z in zs:
  ds=z['dataset'];q=z['query_ordinal'];row=read_row(ds,q);gt=targets[ds][0][q];span=targets[ds][1][q];values={};cfg=p['configurations'][ds];hyp=z['hypotheses'];aa={}
  if not hyp['empty']:
   aa=dict(positions=np.array(hyp['positions']),targets=hyp['targets'].numpy(),mask=hyp['mask'].numpy(),lw=hyp['logw'].numpy().astype(float),tw=hyp['temporal_weights'].numpy())
  for arm in ARMS:
   fit=z['arms'][arm]['fit'];assert torch.equal(fit['before'],z['Source_boxes']);assert fit['active_parameters']==1792;assert all(torch.equal(fit['initial'][k],v)for k,v in z['initial'].items());assert torch.equal(fit['final'],z['arms'][arm]['complete_output']['boxes']);assert fit['config']==cfg;assert not fit['GT_used'];checks+=6
   if arm=='top1_frame':old_audit(fit,load(OLD/'target'/(z['alias']+'.pt'))['expert']);mathchecks+=1
   elif not hyp['empty']:
    for rr in fit['rounds']:
     if arm=='multi_direct':continue
     samples=rr['samples'].numpy();mu=rr['old_mean'].numpy();offsets=rr['offsets'].numpy();lw=rr['logw'].numpy();ids=[z['frame_ids'][i]for i in fit['policy_positions']];K=covariance(ids,torch.device('cpu'),torch.float64,arm=='multi_tube').numpy();inv=np.linalg.inv(K);logdet=np.linalg.slogdet(K)[1];res=samples[:,None]-mu[None,None]-offsets[None];quad=np.einsum('mhnd,nk,mhkd->mh',res,inv,res)/cfg['sigma']**2;lp=logsumexp(lw[None]-.5*(quad+4*(len(mu)*np.log(2*np.pi*cfg['sigma']**2)+logdet)),axis=1);err=float(abs(lp-rr['log_density'].numpy()).max());maxdensity=max(maxdensity,err);assert err<1e-6
     # Independent geometry and hypothesis-first marginalization of every action.
     full=fit['path'][rr['step']]['boxes'].double().numpy()[None].repeat(32,axis=0);full[:,fit['policy_positions']]=1/(1+np.exp(-samples));pred=full[:,None,aa['positions']];ref=aa['targets'][None];px=np.concatenate((pred[...,:2]-pred[...,2:]/2,pred[...,:2]+pred[...,2:]/2),-1);tx=np.concatenate((ref[...,:2]-ref[...,2:]/2,ref[...,:2]+ref[...,2:]/2),-1);inter=np.maximum(0,np.minimum(px[...,2:],tx[...,2:])-np.maximum(px[...,:2],tx[...,:2])).prod(-1);pa=np.maximum(0,px[...,2:]-px[...,:2]).prod(-1);ta=np.maximum(0,tx[...,2:]-tx[...,:2]).prod(-1);iou=inter/np.maximum(pa+ta-inter,1e-7);compat=(iou*aa['mask']*aa['tw']).sum(-1);indreward=cfg['tau']*logsumexp(aa['lw']+compat/cfg['tau'],axis=-1);assert abs(indreward-rr['rewards'].numpy()).max()<2e-7;mathchecks+=32
     w=rr['weights'].numpy();re=rr['rewards'].numpy();expected=np.exp(re/cfg['tau']-logsumexp(re/cfg['tau']));assert abs(expected-w).max()<1e-12;loss=-((w-1/32)*lp).sum()/len(mu);assert abs(loss-rr['loss'])<1e-7;mathchecks+=3
   if arm!='top1_frame' and not hyp['empty']:
    # Verify original Adam algebra from saved actual decoder gradients.
    init=torch.cat([v.reshape(-1)for v in fit['initial'].values()]).double();m=torch.zeros_like(init);v=torch.zeros_like(init);step=0;delta=torch.zeros_like(init)
    for rr in fit['rounds']:
     if not rr.get('updated'):continue
     step+=1;g=rr['gradient'].double();m=.9*m+.1*g;v=.999*v+.001*g*g;expected=-cfg['lr']*(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8);actual=rr['parameter_delta'].double();assert float((expected-actual).abs().max())<2e-5;delta+=actual;mathchecks+=1792
    final=torch.cat([v.reshape(-1)for v in fit['state'].values()]).double();assert float((final-init-delta).abs().max())<2e-6
   cost[ds+'_'+arm].append(z['compute']['arm_seconds'][arm])
  for arm,boxes in [('Source',z['Source_boxes'])]+[(a,z['arms'][a]['fit']['final'])for a in ARMS]:
   ind=DenseTube(boxes.numpy(),row,gt,span,clip=ds=='hc2');values[arm]={}
   for temporal,iv in [('Native',z['Native_interval']),('MBR',z['MBR_interval'])]:
    met=official(boxes.numpy(),row,gt,span,iv,ds);assert max(abs(met[k]-ind.score(iv)[k])for k in ['v','t','s'])<2e-10;checks+=3;met.update(R03=float(met['v']>.3),R05=float(met['v']>.5));values[arm][temporal]=met
   positions=load(OLD/'target'/(z['alias']+'.pt'))['expert']['positions4'];obs=[z['frame_ids'][x]for x in positions];isnative=np.isin(ind.fids,z['frame_ids']);isobs=np.isin(ind.fids,obs)
   values[arm]['geometry']={k:dict(count=int(mask.sum()),sIoU=float(ind.iou[mask].mean())if mask.any()else None)for k,mask in [('observed',isobs),('unobserved_decoder',isnative&~isobs),('interpolated',~isnative)]}
  rows.append(dict(source_alias=z['alias'],dataset=ds,values=values))
  original=load(OLD/'target'/(z['alias']+'.pt'));rawcounts=[len(o['probe']['boxes'])for _,o in sorted(original['expert']['observations'].items())]
  hypmeta.append(dict(source_alias=z['alias'],dataset=ds,empty=hyp['empty'],cached_pool_counts=rawcounts,candidate_filter='finite valid geometry, target score>=.35, strict invertible box chart; no GT selection',candidate_counts=hyp.get('candidate_counts',[]),paths=hyp.get('paths',[]),evidence_weights=hyp['logw'].exp().tolist()if'logw'in hyp else [],actual_identity_annotation='unknown',all_candidate_coordinates_private=True,gradient_calls={a:z['arms'][a]['fit']['gradient_calls']for a in ARMS}))
 groups={d:{t:{a:{k:float(np.mean([r['values'][a][t][k]for r in rows if r['dataset']==d]))for k in ['v','t','s','R03','R05']}for a in ['Source',*ARMS]}for t in ['Native','MBR']}for d in ['hc2','vidstg']};pairs={}
 for d in groups:
  pairs[d]={}
  for t in ['Native','MBR']:
   pairs[d][t]={}
   for a,b in [(x,'Source')for x in ARMS]+[('multi_frame','top1_frame'),('multi_tube','multi_frame'),('multi_tube','multi_direct')]:
    rr=[r for r in rows if r['dataset']==d];diff=np.array([r['values'][a][t]['v']-r['values'][b][t]['v']for r in rr])*100;rg=np.random.default_rng(20261018);ci=np.quantile(diff[rg.integers(0,len(diff),(10000,len(diff)))].mean(1),[.025,.975]);pairs[d][t][a+'-minus-'+b]=dict(mean_pp=float(diff.mean()),CI95=ci.tolist(),positive=int((diff>1e-9).sum()),negative=int((diff< -1e-9).sum()),neutral=int((abs(diff)<=1e-9).sum()),per_source_signed_pp={r['source_alias']:float(x)for r,x in zip(rr,diff)})
 geometry_summary={d:{a:{s:dict(valid_sources=sum(r['values'][a]['geometry'][s]['sIoU']is not None for r in rows if r['dataset']==d),mean_sIoU=float(np.mean([r['values'][a]['geometry'][s]['sIoU']for r in rows if r['dataset']==d and r['values'][a]['geometry'][s]['sIoU']is not None]))if any(r['values'][a]['geometry'][s]['sIoU']is not None for r in rows if r['dataset']==d)else None)for s in ['observed','unobserved_decoder','interpolated']}for a in ['Source',*ARMS]}for d in groups}
 PUB.mkdir(parents=True,exist_ok=True);put(PUB/'RESULTS.json',dict(status='complete',count=32,groups=groups,geometry_summary=geometry_summary,paired=pairs,rows=rows,cohort='previously exposed rollback/critic16/domain; NOT original Temporal32/domain',all_negative_neutral_unknown_retained=True,not_official_or_confirmation=True))
 put(PUB/'HYPOTHESES.json',dict(status='complete',count=32,rows=hypmeta,association_weights_not_correctness_probabilities=True,no_new_identity_semantic_labels=True))
 costs={k:dict(count=len(v),total_seconds=sum(v),mean_seconds=sum(v)/len(v))for k,v in cost.items()};put(PUB/'COST.json',dict(status='complete',count=32,GPU_lease_seconds=bar['GPU_lease_seconds'],Source_decode_association_seconds=sum(z['compute']['capture_decode_association_seconds']for z in zs),fits=costs,new_DINO_calls=0,new_Qwen_calls=0,new_models_trained=0,timing_scope='cached-observation full fresh Source capture + fits and complete readback; historical detector/pixel acquisition/extra root checks excluded; NOT fresh deployment E2E',full_deployment_latency=None))
 put(PUB/'VERIFICATION.json',dict(status='pass',count=32,independent_metric_and_state_checks=checks,independent_original_and_mixture_math_checks=mathchecks,maximum_log_density_error=maxdensity,all32_allarms_sealed_before_new_GT_scoring=True,all_final_boxes_actual_complete_Student_outputs=True,initial_Source_equal_all_arms=True,QR_LN_Adam_episodic=True,GT_used_for_adaptation=False))
 lines=['# Matched identity-hypothesis trajectory OPD32','', 'Fixed previously exposed16 parents/domain, same frozen Source initial1792 coordinates and domain optimizer budgets. No inherited LN/writeback, additional DINO, Qwen, diffusion or trained association head. All32×4 complete final Student outputs sealed before target scoring. Historical online P7-A is not a matched baseline.','', '| Domain | Arm | Native vIoU % | MBR vIoU % | Native tIoU % | Native sIoU % |','|---|---|---:|---:|---:|---:|']
 for d in groups:
  for a in ['Source',*ARMS]:
   m=groups[d]['Native'][a];lines.append(f'| {d} | {a} | {100*m["v"]:.6f} | {100*groups[d]["MBR"][a]["v"]:.6f} | {100*m["t"]:.6f} | {100*m["s"]:.6f} |')
 lines+=['','## What changes and what remains uncertain','', 'Top1 uses original framewise OPD, original admission/reward/seed. Multi-frame changes to cached association hypotheses, Native event-membership weighting and joint cross-anchor reward weights. Thus that contrast is a reference/feedback bundle, not a pure extra-box-count effect. Multi-tube shares exactly the latter feedback, adds smooth correlated complete-tube proposals and an exact mixture policy containing distinct interpolated identity paths. It is a policy change, not a dense-label claim. No identity coordinate mean or per-frame best-identity reward.','', 'Same-evidence Direct maximizes the same differentiable marginalized compatibility with the same state, optimizer and maximum rounds; it can have zero gradient for distant disjoint boxes. Its failure alone would not prove generic OPD superiority over every direct objective. Association uses cached query-conditioned Source ROI appearance and DINO evidence. Coherent paths can still all identify the wrong person, and Top3 may omit the true object. Identity correctness remains unknown without pixel annotation. Outside-event evidence remains for association; event-membership weights geometry only and DINO never chooses boundaries.','', 'Native time is primary; identical no-Actionness MBR time is secondary in all arms. MBR is a standard metric-aware readout, not new semantic supervision. Temporal64 uses a different larger historical cohort and must not be joined into this32 as a factorial population.','', 'All signed per-source effects, unobserved actual decoder-frame geometry and damage tails are retained in RESULTS. Confidence intervals are paired parent bootstrap10000, unadjusted. These are development measurements, not new independent confirmation or full-scale evidence. Sparse reward and structured actions do not guarantee actual identity, event semantics or unobserved geometry.','', 'Actual cached acquisition/fit costs are supplied; fresh deployment latency is unknown. No outcome automatically promotes the candidate, restarts an old queue or triggers parameter/temperature search.']
 (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n');put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True));put(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,math_checks=mathchecks,metric_state_checks=checks,maximum_log_density_error=maxdensity));print(json.dumps(groups,indent=2))
def render():
 import numpy as np,torch
 from PIL import Image,ImageDraw
 from scripts.run_decota_paper_main_v1 import read_row,frames_for
 p=read(BASE/'PLAN_LOCK.json');review=[]
 rr=read(PUB/'RESULTS.json')['rows'];selected=[]
 for ds in ['hc2','vidstg']:
  first=next(x for x in p['records']if x['dataset']==ds);worst=min((x for x in rr if x['dataset']==ds),key=lambda x:x['values']['multi_tube']['Native']['v']-x['values']['Source']['Native']['v']);last=next(x for x in p['records']if x['alias']==worst['source_alias']);selected.extend([(ds,'fixed_first',first),(ds,'posthoc_worst_multitube_harm',last)])
 for ds,selection,r in selected:
  z=load(BASE/'predictions'/(r['alias']+'.pt'));old=load(OLD/'target'/(r['alias']+'.pt'));row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,collections.OrderedDict());obs=old['expert']['positions4'];un=[i for i in np.linspace(0,len(ids)-1,8,dtype=int)if i not in obs][:4];pos=obs+un;canvas=Image.new('RGB',(320*5,220*len(pos)),'white');draw=ImageDraw.Draw(canvas)
  for j,i in enumerate(pos):
   for c,a in enumerate(['Source',*ARMS]):
    im=Image.fromarray(frames[i]).convert('RGB').resize((320,190));dr=ImageDraw.Draw(im);b=z['Source_boxes'][i]if a=='Source'else z['arms'][a]['fit']['final'][i];x,y,w,h=map(float,b);dr.rectangle(((x-w/2)*320,(y-h/2)*190,(x+w/2)*320,(y+h/2)*190),outline='red',width=3);canvas.paste(im,(c*320,j*220));draw.text((c*320+4,j*220+192),f'{a} {"observed"if i in obs else"unobserved decoder"}',fill='black')
  f=BASE/'root_review'/(ds+'_'+selection+'.png');f.parent.mkdir(exist_ok=True);canvas.save(f);review.append(dict(dataset=ds,selection=selection,source_alias=r['alias'],path=str(f),positions=len(pos),cells=5*len(pos),sha256=sha(f),public=False))
 put(BASE/'ROOT_VIEW_INPUTS.json',dict(status='ready_for_actual_view',files=review));print(json.dumps(review))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['score','render']);globals()[p.parse_args().stage]()
