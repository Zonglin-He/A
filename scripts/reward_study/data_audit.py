"""Read-only CPU diagnosis of sealed HC128 layout trial; no decoder or model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json,sys,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
torch.set_num_threads(1)
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve(); TRIAL=ROOT/'outputs/layout_reward_dev'
sys.path.insert(0,str(ROOT))
from stvg.geometry import local_cost

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def arr(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def finite(x): return np.asarray(x,dtype=float)[np.isfinite(x)]
def summary(x):
 x=finite(x)
 return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9))) if len(x) else dict(n=0)
def xy(b): return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
def iou(b,g):
 b=np.maximum(xy(np.asarray(b)),0);g=np.asarray(g)
 a=np.maximum(b[...,2:]-b[...,:2],0).prod(-1);ag=np.maximum(g[...,2:]-g[...,:2],0).prod(-1)
 inter=np.maximum(np.minimum(b[...,2:],g[...,2:])-np.maximum(b[...,:2],g[...,:2]),0).prod(-1)
 return inter/np.maximum(a+ag-inter,1e-15)
def ranks(x,y):
 x=rankdata(x,axis=-1);y=rankdata(y,axis=-1);x-=x.mean(-1,keepdims=True);y-=y.mean(-1,keepdims=True)
 den=np.sqrt((x*x).sum(-1)*(y*y).sum(-1));return np.divide((x*y).sum(-1),den,out=np.full_like(den,np.nan),where=den>0)
def chart(b): return np.concatenate((b[...,:2],np.log(np.maximum(b[...,2:],1e-20))),-1)
def cosine(a,b):
 den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
 return np.divide((a*b).sum(-1),den,out=np.full_like(den,np.nan),where=den>1e-12)

seal=read(TRIAL/'scope_hc2/sealed.json');assert seal['queries']==128 and seal['GT_read'] is False
manifest=ROOT/'data/evaluation/development.jsonl'; assert sha(manifest)==seal['manifest_sha256']
records=[json.loads(l) for l in manifest.read_text().splitlines() if l and json.loads(l)['dataset']=='hc2'];assert len(records)==128
# Validate all NPZ, receipt, trace, and student-cache bindings before loading labels.
for r in records:
 k=r['id'];s=seal['files'][k]; p=TRIAL/s['path']; assert sha(p)==s['sha256'] and sha(p.with_suffix('.json'))==s['receipt_sha256']
 receipt=read(p.with_suffix('.json'));assert sha(TRIAL/'traces'/f'{k}.pt')==receipt['trace_sha256']
 sc=read(TRIAL/'student_cache'/f'{k}.json');assert sha(TRIAL/'student_cache'/f'{k}.pt')==sc['sha256'] and sc['fingerprint']==seal['fingerprint']
labels_path=ROOT/'data/evaluation/development_labels.jsonl'; labels_sha=sha(labels_path)
labels={x['id']:x for x in map(json.loads,labels_path.read_text().splitlines()) if x['dataset']=='hc2'}
metrics={x['id']:x for x in read(TRIAL/'scope_hc2/metrics.json')['rows']}
query=[];anchors=[];all_ranks=[];all_geom_ranks=[];all_gain=[];all_geom_gain=[];all_oracle=[];all_top=[];all_gt_best_r=[];all_ess=[];all_reward_std=[];all_cor_size=[];all_cor_center=[];step_paths=[];neighbor=[];distant=[]
target_cases={36,84,125,126,120}
for j,r in enumerate(records):
 k=r['id'];g=labels[k];meta=r['metadata'];wh=np.array([meta['width'],meta['height']]*2)
 gt={int(f):np.array(b,dtype=float)/wh for f,b in g['boxes'].items()}
 c=torch.load(TRIAL/'student_cache'/f'{k}.pt',map_location='cpu',weights_only=False);tr=torch.load(TRIAL/'traces'/f'{k}.pt',map_location='cpu',weights_only=False)
 with np.load(TRIAL/'predictions'/f'{k}.npz') as z: pred={n:z[n].copy() for n in z.files}
 ids=pred['frame_ids'];pos=pred['positions8'];teacher=arr(c['teacher']['boxes']);mask=arr(c['teacher']['layout_mask']);F=arr(c['teacher']['layout_features']); h,w=F.shape[-2:]
 valid_hw=np.array([[np.any(~m,axis=1).sum(),np.any(~m,axis=0).sum()] for m in mask]);feature_wh=valid_hw[:,::-1]
 actual=np.array([int(f) in gt for f in ids]);observed=np.isin(np.arange(len(ids)),pos);have=np.array([int(ids[p]) in gt for p in pos])
 idx=np.flatnonzero(have);agt=np.array([gt[int(ids[pos[a]])] for a in idx]) if len(idx) else np.zeros((0,4))
 teacher_iou=iou(teacher[idx],agt) if len(idx) else np.array([])
 gt_normal=np.array([gt[int(f)] for f in ids[actual]])
 quality=float(teacher_iou.mean()) if len(idx) else np.nan
 f=F.transpose(0,2,3,1); f=f/np.maximum(np.linalg.norm(f,axis=-1,keepdims=True),1e-8)
 neighbor.append(float(((f[:,:,:-1]*f[:,:,1:]).sum(-1).mean()+(f[:,:-1]*f[:,1:]).sum(-1).mean())/2))
 flat=f.reshape(8,-1,256); n=flat.shape[1];a=np.arange(0,n,max(1,n//32))[:32];b=(a+n//2)%n;distant.append(float((flat[:,a]*flat[:,b]).sum(-1).mean()))
 q=dict(id=k,public_id=f'H{j+1:03}',GT_anchors=int(have.sum()),total_anchors=8,teacher_IoU=quality,GT_relative_area=float(np.maximum(gt_normal[:,2:]-gt_normal[:,:2],0).prod(-1).mean()),teacher_feature_min_extent=float((teacher[:,2:]*feature_wh).min()),teacher_feature_extent_mean=(teacher[:,2:]*feature_wh).mean(0).tolist(),feature_neighbor_cos=neighbor[-1],feature_distant_position_cos=distant[-1],arms={},ranking={},metrics=metrics[k]['values'])
 for name,mm in [('E_Ref','WithoutOPD'),('GeometryOPD','GeometryOPD'),('LayoutOPD','Full'),('LayoutDirect','LayoutDirect')]:
  bs=pred[name];e=pred['E_Ref'];shift=chart(bs)-chart(e);diag=np.concatenate((e[:,2:],np.ones((len(e),2))),-1);shift/=diag
  ai=iou(bs[pos[idx]],agt) if len(idx) else np.array([])
  every=np.full(len(ids),np.nan);every[actual]=iou(bs[actual],gt_normal)
  q['arms'][name]=dict(observed_IoU=summary(ai),normalized_center_shift=summary(np.linalg.norm(shift[:,:2],axis=1)),log_size_shift=summary(np.linalg.norm(shift[:,2:],axis=1)),anchor_chart_l2_to_teacher=summary(np.linalg.norm((chart(bs[pos])-chart(teacher))/np.concatenate((teacher[:,2:],np.ones((8,2))),-1),axis=1)),actual_IoU=every.tolist())
 for name in ['LayoutOPD','LayoutDirect']:
  path=tr[name]['path'];reward0=arr(path[0]['layout_reward']);reward1=arr(path[-1]['layout_reward'])
  before=arr(path[0]['boxes']);after=arr(path[-1]['boxes']);dR=reward1-reward0
  diou=iou(after[pos[idx]],agt)-iou(before[pos[idx]],agt) if len(idx) else np.array([])
  updates=np.stack([arr(t['state_step']) for t in tr[name]['rounds']]);grads=np.stack([arr(t['gradient']) for t in tr[name]['rounds']]);steps_norm=np.linalg.norm(updates.reshape(40,-1),axis=1)
  q['arms'][name].update(reward_start=reward0.tolist(),reward_end=reward1.tolist(),reward_gain=summary(dR),GT_anchor_IoU_gain=summary(diou),reward_up_IoU_down=int(((dR[idx]>1e-8)&(diou < -1e-8)).sum()),reward_up_count=int((dR[idx]>1e-8).sum()),gradient_norm=summary(np.linalg.norm(grads.reshape(40,-1),axis=1)),state_step_norm=summary(steps_norm),total_state_norm=float(np.linalg.norm(updates.sum(0))),sum_state_step_norm=float(steps_norm.sum()))
  for a in idx:
   bb=before[pos[a]];aa=after[pos[a]];gg=np.r_[(gt[int(ids[pos[a]])][:2]+gt[int(ids[pos[a]])][2:])/2,gt[int(ids[pos[a]])][2:]-gt[int(ids[pos[a]])][:2]]
   gtvec=(chart(gg)-chart(bb))/np.r_[bb[2:],1,1];svec=(chart(aa)-chart(bb))/np.r_[bb[2:],1,1]
   anchors.append(dict(id=k,public_id=q['public_id'],arm=name,anchor=int(a),frame=int(ids[pos[a]]),teacher_IoU=float(iou(teacher[a],agt[list(idx).index(a)])),initial_IoU=float(iou(bb,agt[list(idx).index(a)])),final_IoU=float(iou(aa,agt[list(idx).index(a)])),reward_gain=float(dR[a]),direction_cos=float(cosine(svec,gtvec)),GT_min_feature_extent=float(((gg[2:])*feature_wh[a]).min()),teacher_min_feature_extent=float((teacher[a,2:]*feature_wh[a]).min())))
  # All saved real decoder paths; never selects a stop step online.
  path_rows=[]
  for p in path:
   b=arr(p['boxes']);a_i=iou(b[actual],gt_normal);path_rows.append(dict(step=p['step'],mean_layout_reward=float(arr(p['layout_reward']).mean()),actual_sIoU=float(a_i.mean()),unobserved_sIoU=float(iou(b[actual&~observed],gt_normal[~observed[actual]]).mean()) if (actual&~observed).any() else np.nan,observed_sIoU=float(iou(b[pos[idx]],agt).mean()) if len(idx) else np.nan))
  step_paths.append(dict(id=k,arm=name,path=path_rows));q['arms'][name]['path']=path_rows if j+1 in target_cases else None
 if len(idx):
  cand=np.stack([arr(t['rollout']['boxes']) for t in tr['LayoutOPD']['rounds']])[:,idx]
  rew=np.stack([arr(t['rollout']['rewards']) for t in tr['LayoutOPD']['rounds']])[:,idx]
  weights=np.stack([arr(t['rollout']['weights']) for t in tr['LayoutOPD']['rounds']])[:,idx]
  ess=np.stack([arr(t['rollout']['ESS']) for t in tr['LayoutOPD']['rounds']])[:,idx]
  ciou=iou(cand,agt[None,:,None,:]);rr=ranks(rew,ciou)
  geom=-arr(local_cost(torch.as_tensor(cand),torch.as_tensor(teacher[idx])[None,:,None,:]));gr=ranks(geom,ciou)
  gw=np.exp((geom-geom.max(-1,keepdims=True))/.14);gw/=gw.sum(-1,keepdims=True)
  gain=(ciou*weights).sum(-1)-ciou.mean(-1);ggain=(ciou*gw).sum(-1)-ciou.mean(-1);top=np.take_along_axis(ciou,rew.argmax(-1)[...,None],-1).squeeze(-1)-ciou.mean(-1)
  oracle=ciou.max(-1)-ciou.mean(-1);rstd=rew.std(-1)
  gt_best_r=np.take_along_axis(rew,ciou.argmax(-1)[...,None],-1).squeeze(-1)-rew.mean(-1)
  all_ranks.extend(rr.flatten());all_geom_ranks.extend(gr.flatten());all_gain.extend(gain.flatten());all_geom_gain.extend(ggain.flatten());all_oracle.extend(oracle.flatten());all_top.extend(top.flatten());all_gt_best_r.extend(gt_best_r.flatten());all_ess.extend(ess.flatten());all_reward_std.extend(rstd.flatten())
  delta=chart(cand)-np.stack([arr(t['rollout']['proposal_mean']) for t in tr['LayoutOPD']['rounds']])[:,idx,None,:]
  size_delta=delta[...,2:].sum(-1);center_to_teacher=-np.linalg.norm((cand[...,:2]-teacher[idx,None,:2])/teacher[idx,None,2:],axis=-1)
  crsize=ranks(rew,size_delta);crcenter=ranks(rew,center_to_teacher);all_cor_size.extend(crsize.flatten());all_cor_center.extend(crcenter.flatten())
  q['ranking']=dict(layout_spearman=summary(rr),geometry_spearman=summary(gr),layout_weighted_IoU_gain=summary(gain),geometry_reweight_same_candidates_IoU_gain=summary(ggain),oracle_IoU_gain=summary(oracle),layout_top1_IoU_gain=summary(top),layout_reward_std=summary(rstd),layout_ESS=summary(ess),layout_IoU_gain_positive_fraction=float((gain>0).mean()),layout_vs_log_area_spearman=summary(crsize),layout_vs_teacher_center_spearman=summary(crcenter))
 query.append(q)
 del c,tr,F,f,pred;gc.collect()
 if (j+1)%16==0:print('scanned',j+1,flush=True)

def avgpath(arm):
 pp=[r['path'] for r in step_paths if r['arm']==arm];return[{k:float(np.nanmean([p[s][k] for p in pp])) for k in ['step','mean_layout_reward','actual_sIoU','unobserved_sIoU','observed_sIoU']} for s in range(41)]
aggregate=dict(queries=128,anchors_total=1024,anchors_with_GT=sum(q['GT_anchors'] for q in query),queries_without_GT_anchors=sum(q['GT_anchors']==0 for q in query),GT_anchor_count_distribution={str(i):sum(q['GT_anchors']==i for q in query) for i in range(9)},teacher_IoU=summary([a['teacher_IoU'] for a in anchors if a['arm']=='LayoutOPD']),GT_min_feature_extent=summary([a['GT_min_feature_extent'] for a in anchors if a['arm']=='LayoutOPD']),GT_min_feature_extent_lt3_fraction=float(np.mean([a['GT_min_feature_extent']<3 for a in anchors if a['arm']=='LayoutOPD'])),GT_min_feature_extent_lt1_fraction=float(np.mean([a['GT_min_feature_extent']<1 for a in anchors if a['arm']=='LayoutOPD'])),feature_neighbor_cos=summary(neighbor),feature_distant_position_cos=summary(distant),layout_spearman=summary(all_ranks),geometry_spearman_same_candidates=summary(all_geom_ranks),layout_positive_spearman_fraction=float(np.mean(finite(all_ranks)>0)),geometry_positive_spearman_fraction=float(np.mean(finite(all_geom_ranks)>0)),layout_weighted_IoU_gain=summary(all_gain),geometry_same_candidates_weighted_IoU_gain=summary(all_geom_gain),oracle32_IoU_gain=summary(all_oracle),layout_top1_IoU_gain=summary(all_top),layout_reward_std=summary(all_reward_std),layout_ESS=summary(all_ess),layout_reward_vs_log_area_spearman=summary(all_cor_size),layout_reward_vs_teacher_center_spearman=summary(all_cor_center),mean_paths={a:avgpath(a) for a in ['LayoutOPD','LayoutDirect']},arms={})
for arm in ['LayoutOPD','LayoutDirect']:
 a=[x for x in anchors if x['arm']==arm];aggregate['arms'][arm]=dict(anchors_with_GT=len(a),reward_increased=int(sum(x['reward_gain']>1e-8 for x in a)),reward_up_IoU_down=int(sum(x['reward_gain']>1e-8 and x['final_IoU']<x['initial_IoU']-1e-8 for x in a)),IoU_improved=int(sum(x['final_IoU']>x['initial_IoU']+1e-8 for x in a)),GT_direction_cos=summary([x['direction_cos'] for x in a]),GT_direction_positive_fraction=float(np.mean(finite([x['direction_cos'] for x in a])>0)),mean_GT_anchor_gain=summary([x['final_IoU']-x['initial_IoU'] for x in a]),reward_gain=summary([x['reward_gain'] for x in a]))
quality=np.array([q['teacher_IoU'] for q in query]);valid=np.isfinite(quality);edges=np.quantile(quality[valid],[.25,.5,.75]);bins=np.digitize(quality,edges);groups=[]
for group in range(4):
 qq=[q for i,q in enumerate(query) if valid[i] and bins[i]==group];groups.append(dict(group=group+1,n=len(qq),teacher_IoU=summary([q['teacher_IoU'] for q in qq]),layout_minus_geometry_v_pp=float(np.mean([(q['metrics']['Full']['v']-q['metrics']['GeometryOPD']['v'])*100 for q in qq])),layout_minus_E_v_pp=float(np.mean([(q['metrics']['Full']['v']-q['metrics']['WithoutOPD']['v'])*100 for q in qq])),ranking_layout_rho=summary([q['ranking']['layout_spearman'].get('mean',np.nan) for q in qq])))
aggregate['teacher_quality_quartiles']=groups
aggregate['query_correlations']={}
for name,values in [('GT_anchors',[q['GT_anchors'] for q in query]),('teacher_IoU',[q['teacher_IoU'] for q in query]),('GT_relative_area',[q['GT_relative_area'] for q in query]),('mean_teacher_feature_min_extent',[q['teacher_feature_min_extent'] for q in query]),('candidate_ranking',[q.get('ranking',{}).get('layout_spearman',{}).get('mean',np.nan) for q in query])]:
 x=np.array(values);y=np.array([(q['metrics']['Full']['v']-q['metrics']['GeometryOPD']['v'])*100 for q in query]);m=np.isfinite(x)&np.isfinite(y);s=spearmanr(x[m],y[m]);aggregate['query_correlations'][name]=dict(n=int(m.sum()),rho=float(s.statistic),p=float(s.pvalue))
assert labels_sha==sha(labels_path) and not torch.cuda.is_initialized()
out=dict(status='CPU_saved_artifact_only_GT_posthoc',scope='HC128_exposed_development_not_formal_Vid',seal_sha256=sha(TRIAL/'scope_hc2/sealed.json'),labels_sha256=labels_sha,no_model_no_decoder_no_GPU=True,aggregate=aggregate,queries=query,anchors=anchors)
(TRIAL/'research/data_audit.json').write_text(json.dumps(out,indent=2,allow_nan=True)+'\n')
print(json.dumps(aggregate,indent=2))
