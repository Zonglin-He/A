"""Offline fixed-candidate diagnostic of cached feature correspondence controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_correspondence32_v1 import D,OLD,PUB,ARMS,read,put,load,sha,mean
from scripts.score_stvg_opd_p1_v1 import truths
from vg_tta.tastvg_oracle_event5_v1 import box_iou
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap,soft_iou
import numpy as np,torch
from scipy.stats import rankdata

def matched_weights(reward,reference):
 """Same sorted reference weights, assigned by each reward's rank; average ties."""
 out=np.zeros_like(reward,dtype=float)
 for i,(r,ref)in enumerate(zip(reward,reference)):
  idx=np.argsort(r,kind='stable');s=r[idx];w=np.sort(ref);start=0
  while start<len(s):
   end=start+1
   while end<len(s)and s[end]==s[start]:end+=1
   out[i,idx[start:end]]=w[start:end].mean();start=end
 return out

def rankings(reward,w,quality,baseline):
 weighted=(w*quality).sum(1)-baseline;rho=[];pair=[];top=[];ii,jj=np.triu_indices(32,1)
 for r,q in zip(reward,quality):
  a=rankdata(r);b=rankdata(q);aa=a-a.mean();bb=b-b.mean();den=np.linalg.norm(aa)*np.linalg.norm(bb);rho.append(float(aa@bb/den)if den>0 else None)
  diff=q[ii]-q[jj];take=abs(diff)>1e-12;rd=r[ii]-r[jj];pair.append(float(np.mean(np.where(abs(rd[take])<=1e-12,.5,(rd[take]*diff[take]>0).astype(float))))if take.any()else None);top.append(float(q[r==r.max()].mean()-q.mean()))
 return dict(rank_delta=weighted,Spearman=np.array(rho,object),pair_accuracy=np.array(pair,object),top_reward_gain=np.array(top))

def cluster(vals):
 """One anonymous query per parent in this fixed32; retain explicit parent groups."""
 groups=collections.defaultdict(list)
 for parent,v in vals:
  if v is not None:groups[parent].append(v)
 a=np.array([np.mean(v)for v in groups.values()]);out=dict(parents=len(a),mean=float(a.mean())if len(a)else None)
 if len(a):
  rng=np.random.default_rng(20261018);out['CI95']=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()
 return out

def score():
 p=read(D/'PLAN_LOCK.json');bar=read(D/'CONTROL_BARRIER.json');assert bar['count']==32 and not bar['GT_read'];assert all(sha(ROOT/f['path'])==f['sha256']for f in bar['files']);records=p['records'];target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in records if r['dataset']==ds]))for ds in ['hc2','vidstg']};feat=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json');fby={d:{f['query_ordinal']:f for f in fs}for d,fs in feat['rows'].items()};plans={ds:read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')['rows']for ds in target};rows=[];checks=0
 for rec in records:
  z=load(OLD/'predictions'/(rec['alias']+'.pt'));c=load(D/'controls'/(rec['alias']+'.pt'));ds=rec['dataset'];q=rec['query_ordinal'];f=fby[ds][q];row=plans[ds][q];gt=target[ds][0][q];ids=np.asarray(z['frame_ids']);known=np.array([int(i)in gt for i in ids]);observed=np.isin(np.arange(len(ids)),z['planned_positions']);far=np.min(abs(ids[:,None]-ids[z['planned_positions']][None,:]),axis=1)>=.25*(ids[-1]+1-ids[0]);masks=dict(unobserved=known&~observed,far=known&~observed&far,observed=known&observed,all_known=known)
  a=z['actions'].double().numpy();xy=np.concatenate((a[...,:2]-a[...,2:]/2,a[...,:2]+a[...,2:]/2),-1)*np.array([row['input']['width'],row['input']['height']]*2)
  if ds=='hc2':xy=np.maximum(xy,0)
  quality=np.zeros(a.shape[:-1]);gtcell=[]
  for j in np.flatnonzero(known):
   b=np.asarray(gt[int(ids[j])]);quality[j]=box_iou(xy[j],b);bn=b/np.array([row['input']['width'],row['input']['height']]*2);bb=np.r_[(bn[:2]+bn[2:])/2,bn[2:]-bn[:2]];h,w=c['grid'];mask=cell_overlap(torch.tensor(bb)[None],h,w);gtcell.append(dict(observed=bool(observed[j]),width_cells=float(bb[2]*w),height_cells=float(bb[3]*h),area_cells=float(bb[2]*bb[3]*h*w),box_grid_self_iou=float(soft_iou(torch.tensor(bb)[None],mask)[0])))
  baseline=quality.mean(1);stats={};perframe={};reference=z['weights']['interpolation'].double().numpy()
  for arm in ARMS:
   rr=c['rewards'][arm].double().numpy();w=c['weights'][arm].double().numpy();assert abs(w.sum(-1)-1).max()<1e-5;common=matched_weights(rr,reference);assert abs(common.sum(-1)-1).max()<1e-5;v=rankings(rr,w,quality,baseline);vv=rankings(rr,common,quality,baseline);stats[arm]={}
   for name,mask in masks.items():stats[arm][name]=dict(frames=int(mask.sum()),rank_delta=mean(v['rank_delta'][mask]),matched_strength_rank_delta=mean(vv['rank_delta'][mask]),Spearman=mean(v['Spearman'][mask]),pair_accuracy=mean(v['pair_accuracy'][mask]),top_reward_gain=mean(v['top_reward_gain'][mask]))
   perframe[arm]=dict(rank_delta=v['rank_delta'].tolist(),matched_strength_rank_delta=vv['rank_delta'].tolist())
   # Original feedback controls must recover all original per-query numbers.
   if arm in ['real','near_shift','interpolation']:
    oldname='shuffled'if arm=='near_shift'else arm;assert torch.equal(c['rewards'][arm],z['rewards'][oldname]);checks+=1
  # No mixed-query support filtering by GT/reference quality; fixed masks above.
  loo=[]
  for j,code in enumerate(c['anchor_codes']):
   item=dict(slot=int(z['planned_positions'].index(z['admitted_positions'][j])),old_model_assisted_code=code)
   for family,rs in [('real',c['real_leave_one_out']),('interpolation',c['interpolation_leave_one_out'])]:
    reward=rs[j].double().numpy();ew=np.exp((reward-reward.max(-1,keepdims=True))/z['config']['tau']);ew/=ew.sum(-1,keepdims=True);delta=(ew*quality).sum(-1)-baseline;original=np.array(perframe[family]['rank_delta']);item[family]={name:dict(rank_delta=mean(delta[mask]),effect_of_dropping=mean((delta-original)[mask]))for name,mask in masks.items()}
   loo.append(item)
  rows.append(dict(alias=rec['alias'],dataset=ds,parent_alias=f['parent_alias'],target_area_bin=f['bins']['target_area'],motion_bin=f['bins']['speed_sizes_sec'],query_form=f['query_form'],reference_group='WI_present'if'WI'in c['anchor_codes']else'no_WI_label',unknown_unobserved_frames=int((~known&~observed).sum()),grid=c['grid'],anchor_codes=c['anchor_codes'],anchor_stats=c['anchor_stats'],gt_grid_geometry=gtcell,stats=stats,leave_one_anchor_out=loo,gap=c['gap'],aligned_feature_cosine=c['aligned_feature_cosine'],reward_stats=c['reward_stats'],CPU_CUDA_map_probe_max_error=c['cpu_vs_original_CUDA_max_map_error'],nearest_map_replication_invariance=c['upsample_invariance_max_error']))
  print('SCORE',rec['alias'],len(rows),flush=True)
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];out={}
  for support in masks:
   out[support]={arm:{metric:cluster([(r['parent_alias'],r['stats'][arm][support][metric])for r in rr])for metric in ['rank_delta','matched_strength_rank_delta','Spearman','pair_accuracy','top_reward_gain']}for arm in ARMS}
  pairs={}
  contrasts=[('real','near_shift'),('real','far_shift'),('real','uniform_affinity'),('real','interpolation'),('interpolation_raster','interpolation'),('real','interpolation_raster'),('hard_correspondence','real'),('centered_correspondence','real'),('half_grid_correspondence','real'),('real','static_box_masks'),('real','self_match_static')]
  for sup in ['unobserved','far','observed']:
   pairs[sup]={a+'__minus__'+b:{met:cluster([(r['parent_alias'],r['stats'][a][sup][met]-r['stats'][b][sup][met]if r['stats'][a][sup][met]is not None and r['stats'][b][sup][met]is not None else None)for r in rr])for met in ['rank_delta','matched_strength_rank_delta']}for a,b in contrasts}
  lout={}
  for code in ['WI','CI','CP','MX','UN']:
   lout[code]={family:{support:cluster([(r['parent_alias'],mean([a[family][support]['effect_of_dropping']for a in r['leave_one_anchor_out']if a['old_model_assisted_code']==code]))for r in rr])for support in ['unobserved','far']}for family in ['real','interpolation']}
  mixed=[]
  for r in rr:
   bad=[a for a in r['leave_one_anchor_out']if a['old_model_assisted_code']=='WI'];good=[a for a in r['leave_one_anchor_out']if a['old_model_assisted_code']in ['CI','CP']]
   if bad and good:mixed.append(dict(alias=r['alias'],parent_alias=r['parent_alias'],drop_WI_minus_drop_CI_CP={fa:{su:mean([a[fa][su]['effect_of_dropping']for a in bad])-mean([a[fa][su]['effect_of_dropping']for a in good])if mean([a[fa][su]['effect_of_dropping']for a in bad])is not None and mean([a[fa][su]['effect_of_dropping']for a in good])is not None else None for su in ['unobserved','far']}for fa in ['real','interpolation']}))
  geom=[g for r in rr for g in r['gt_grid_geometry']if not g['observed']];ast=[a for r in rr for a in r['anchor_stats']]
  groups[ds]=dict(results=out,paired=pairs,anchor_leave_one_out=lout,mixed_anchor_queries=mixed,mixed_query_paired={fa:{su:cluster([(r['parent_alias'],r['drop_WI_minus_drop_CI_CP'][fa][su])for r in mixed])for su in ['unobserved','far']}for fa in ['real','interpolation']},resolution=dict(evaluable_GT_frames=len(geom),fraction_GT_width_lt1cell=mean([g['width_cells']<1 for g in geom]),fraction_GT_height_lt1cell=mean([g['height_cells']<1 for g in geom]),mean_GT_width_cells=mean([g['width_cells']for g in geom]),mean_GT_height_cells=mean([g['height_cells']for g in geom]),mean_GT_same_rectangle_grid_self_IoU=mean([g['box_grid_self_iou']for g in geom]),mean_DINO_raster_self_IoU=mean([a['raster_self_iou']for a in ast])),attention={k:mean([a[k]for a in ast])for k in ['normalized_entropy','effective_reference_patches','reference_cells','top1_attention','self_diagonal_mass','self_mask_MAE','self_mask_mass_ratio','feature_cosine_all_patch_mean']},feature_similarity={k:mean([r['aligned_feature_cosine'][k]for r in rr])for k in ['near_shift','far_shift']})
 PUB.mkdir(parents=True,exist_ok=True)
 private=dict(status='complete',count=32,groups=groups,rows=rows,original_qualification_unchanged=True,GT_offline_only=True,new_models=0,new_training=0,GPU_seconds=0,CPU_control_wall_seconds=bar['CPU_wall_seconds'],retrospective_not_independent_confirmation=True)
 put(D/'POSTSEAL_RESULTS_PRIVATE.json',private)
 safe=[]
 for r in rows:
  out={k:v for k,v in r.items()if k!='gt_grid_geometry'}
  out['anchor_stats']=[{k:v for k,v in a.items()if k not in ['anchor_position','reference_width_cells','reference_height_cells','mask_mean']}for a in r['anchor_stats']]
  safe.append(out)
 put(PUB/'RESULTS.json',dict(**{k:v for k,v in private.items()if k!='rows'},rows=safe,per_frame_GT_geometry_excluded=True));put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,original_control_checks=checks,new_models=0,new_training=0,GPU_seconds=0));print('SCORED32',flush=True)
if __name__=='__main__':score()
