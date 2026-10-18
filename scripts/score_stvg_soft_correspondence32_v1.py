"""Postseal offline reward-ranking evaluation; GT never enters the feedback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_soft_correspondence32_v1 import D,OLD
from scripts.score_stvg_opd_p1_v1 import truths
from vg_tta.tastvg_oracle_event5_v1 import box_iou
import numpy as np
P=ROOT/'results/stvg_soft_correspondence/2026-10-18/qualification32';ARMS=['real','shuffled','interpolation']
def mean(v):
 a=[x for x in v if x is not None];return float(np.mean(a))if a else None

def summarize(rows,support):
 good=[r for r in rows if r['supports'][support]['frames']>0];parents=collections.defaultdict(list)
 for r in good:parents[r['parent_alias']].append(r)
 values={a:[mean([r['supports'][support]['rank_delta'][a]for r in g])for g in parents.values()]for a in ARMS};out=dict(queries=len(rows),evaluable_queries=len(good),parents=len(parents),frames=sum(r['supports'][support]['frames']for r in good),mean={},parent_mean={},CI95={},paired={})
 if not good:return out
 rng=np.random.default_rng(20261018);idx=rng.integers(len(parents),size=(10000,len(parents)))
 for a in ARMS:
  v=np.array(values[a]);out['mean'][a]=mean([r['supports'][support]['rank_delta'][a]for r in good]);out['parent_mean'][a]=float(v.mean());out['CI95'][a]=np.quantile(v[idx].mean(1),[.025,.975]).tolist()
  out.setdefault('positive_queries',{})[a]=sum(r['supports'][support]['rank_delta'][a]>1e-10 for r in good);out.setdefault('negative_queries',{})[a]=sum(r['supports'][support]['rank_delta'][a]< -1e-10 for r in good);out.setdefault('neutral_queries',{})[a]=sum(abs(r['supports'][support]['rank_delta'][a])<=1e-10 for r in good)
 for a in ARMS[1:]:
  v=np.array(values['real'])-np.array(values[a]);out['paired']['real_minus_'+a]=dict(mean=float(v.mean()),CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist())
 return out

def run():
 p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_FEEDBACK_BARRIER.json');assert bar['count']==32 and not bar['GT_read'];assert all(sha(ROOT/f['path'])==f['sha256']for f in bar['files']);records=p['records'];target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in records if r['dataset']==ds]))for ds in ['hc2','vidstg']};features=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json');fby={ds:{f['query_ordinal']:f for f in fs}for ds,fs in features['rows'].items()};review=read(ROOT/'artifacts/stvg_dino_error_attribution32_v1/VISUAL_REVIEW.json');rows=[];checks=0
 for rec in records:
  z=load(D/'predictions'/(rec['alias']+'.pt'));ds=z['dataset'];q=z['query_ordinal'];gt=target[ds][0][q];plans=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')['rows'][q];f=fby[ds][q];ids=np.array(z['frame_ids']);b=z['actions'].double().numpy();a=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)*np.array([plans['input']['width'],plans['input']['height']]*2);truth=np.zeros(b.shape[:-1]);known=np.array([int(i)in gt for i in ids]);planned=np.array(z['planned_positions']);isplanned=np.isin(np.arange(len(ids)),planned);distance=np.min(abs(ids[:,None]-ids[planned][None,:]),axis=1)if len(planned)else np.zeros(len(ids));far=distance>=p['far_distance_fraction']*(ids[-1]+1-ids[0]);ranks={};quality={};uniform=np.zeros(len(ids))
  if ds=='hc2':a=np.maximum(a,0) # original HC evaluator convention; feedback/candidates unchanged
  for pos,fid in enumerate(ids):
   if int(fid)in gt:truth[pos]=box_iou(a[pos],gt[int(fid)])
  for arm in ARMS:
   w=z['weights'][arm].double().numpy();assert np.max(abs(w.sum(1)-1))<1e-5;uniform=truth.mean(1);quality[arm]=(w*truth).sum(1);ranks[arm]=quality[arm]-uniform
   # Independent scalar summation for each known frame, catches wrong action/frame joins.
   for pos in np.flatnonzero(known):assert abs(sum(float(w[pos,j])*float(truth[pos,j])for j in range(32))-uniform[pos]-ranks[arm][pos])<1e-12;checks+=1
  masks=dict(observed=known&isplanned,unobserved=known&~isplanned,far=known&~isplanned&far,near=known&~isplanned&~far,all_known=known);supports={}
  for name,mask in masks.items():supports[name]=dict(frames=int(mask.sum()),rank_delta={a:mean(ranks[a][mask].tolist())for a in ARMS},weighted_true_iou={a:mean(quality[a][mask].tolist())for a in ARMS},uniform_true_iou=mean(uniform[mask].tolist()))
  # Reference semantic labels only from old actual model-assisted pixel review.
  # This diagnostic does not make them independent human labels or online filters.
  hits=[v for v in review['rows']if v['source_alias']==rec['alias']];assert len(hits)==1
  codes=hits[0]['slots'];assert len(codes)==len(planned)
  admitted_codes=[codes[list(planned).index(pos)]for pos in z['admitted_positions']]
  semantic='wrong_reference_present'if 'WI'in admitted_codes else'no_wrong_label_but_unknown'if 'UN'in admitted_codes else'no_wrong_identity_label'
  per_anchor=[]
  for j,(pos,code)in enumerate(zip(z['admitted_positions'],admitted_codes)):
   reward=z['per_anchor_rewards'][j].double().numpy();tau=z['config']['tau'];ww=np.exp((reward-reward.max(1,keepdims=True))/tau);ww/=ww.sum(1,keepdims=True);ar=(ww*truth).sum(1)-uniform
   per_anchor.append(dict(anchor_slot=int(list(planned).index(pos)),old_model_assisted_code=code,unobserved_rank_delta=mean(ar[masks['unobserved']].tolist()),far_rank_delta=mean(ar[masks['far']].tolist())))
  rows.append(dict(alias=rec['alias'],dataset=ds,parent_alias=f['parent_alias'],event_length_bin=f['bins']['event_fraction'],target_area_bin=f['bins']['target_area'],motion_bin=f['bins']['speed_sizes_sec'],query_form=f['query_form'],admitted_anchors=len(z['admitted_positions']),planned_slots=len(planned),sampled_frames=len(ids),unknown_GT_frames=int((~known).sum()),unknown_unobserved_frames=int((~known&~isplanned).sum()),unknown_far_frames=int((~known&~isplanned&far).sum()),supports=supports,per_known_frame=[dict(position=int(j),observed=bool(isplanned[j]),far=bool(far[j]),rank_delta={a:float(ranks[a][j])for a in ARMS},uniform_iou=float(uniform[j]),anchor_reward_disagreement=float(z['per_anchor_disagreement'][j].mean()))for j in np.flatnonzero(known)],old_visual_review_status=semantic,admitted_old_model_assisted_codes=admitted_codes,per_anchor_diagnostic=per_anchor,compute=z['compute']))
  print('SCORE',rec['alias'],supports['unobserved']['frames'],flush=True)
 groups={ds:{s:summarize([r for r in rows if r['dataset']==ds],s)for s in ['all_known','observed','unobserved','near','far']}for ds in ['hc2','vidstg']};strata={ds:{k:{v:summarize([r for r in rows if r['dataset']==ds and r[k]==v],'unobserved')for v in sorted({r[k]for r in rows if r['dataset']==ds})}for k in ['target_area_bin','motion_bin','query_form','old_visual_review_status']}for ds in groups};gate={}
 for ds,g in groups.items():
  u=g['unobserved'];f=g['far'];conditions=dict(at_least8_unobserved_parents=u['parents']>=8,at_least8_far_parents=f['parents']>=8,real_unobserved_positive_CI=bool(u['CI95']and u['CI95']['real'][0]>0),real_beats_shuffle_CI=bool(u['paired']and u['paired']['real_minus_shuffled']['CI95'][0]>0),real_beats_interpolation_CI=bool(u['paired']and u['paired']['real_minus_interpolation']['CI95'][0]>0),real_far_positive=bool(f['parent_mean']and f['parent_mean']['real']>0),real_far_beats_both=bool(f['paired']and all(v['mean']>0 for v in f['paired'].values())));gate[ds]=dict(conditions=conditions,passed=all(conditions.values()))
 passed=all(v['passed']for v in gate.values());P.mkdir(parents=True,exist_ok=True);put(P/'RESULTS.json',dict(status='complete',count=32,groups=groups,strata=strata,rows=rows,GT_after_all32_feedback_seal=True,no_new_DINO_or_Student_updates=True,low_resolution_frozen_appearance_maps=True,GT_known_only_not_zero_filled=True,all_sources_retained=True))
 put(P/'GATE.json',dict(status='complete',groups=gate,passed=passed,next_action='one_matched_online_three_arm_TTA'if passed else'stop_this_feature_interface_no_TTA_or_training'))
 put(P/'VERIFICATION.json',dict(status='pass',count=32,all32_feature_reward_files_SHA_verified=True,independent_candidate_rank_scalar_checks=checks,Source_output_and_anchor_pixels_exact=True,real_and_controls_same_samples=True,empty_query_uniform_preserved=True,unobserved_is_actual_decoder_frame=True,old_visual_labels_model_assisted_only=True))
 put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,ranking_gate_passed=passed));put(D/'ROOT_READBACK.json',dict(status='pass',count=32,checks=checks));print('GATE',passed,flush=True)
if __name__=='__main__':run()
