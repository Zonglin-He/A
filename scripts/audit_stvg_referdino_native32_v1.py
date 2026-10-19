"""Independent native choice, E/interpolation, metric and timing readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referdino_native32_v1 import D,P,PRE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import numpy as np,torch
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');gate=read(P/'TIMING_CORRECTION_PUBLIC.json');assert gate['status']=='pass'and bar['count']==32 and not torch.cuda.is_initialized();z=read(P/'RESULTS_PUBLIC.json');start=time.perf_counter();count=dict(selection=0,position_rules=0,old_RGB_hashes=0,E_reconstruction=0,interpolation=0,metrics=0,actual_nodes=0,teacher_anchor=0,bootstraps=0);maximum=dict(geometry=0.,metric=0.,selected_score=0.)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in plans};cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');ff=read(P/'FRAME_DIAGNOSTICS_PUBLIC.json')['rows']
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));b=load(D/'predictions'/(a+'.pt'));row=plans[ds][q];dense,spans,_=gt[ds];pub=next(x for x in z['rows']if x['alias']==a);n=b['native'];scores=n['native_query_scores'].numpy();assert n['selected_query']==int(scores.argmax());se=abs(float(n['selected_frame_scores'].mean())-float(scores[n['selected_query']]));assert se<1e-6;maximum['selected_score']=max(maximum['selected_score'],se);assert n['native_same_track_all8']and n['mask_summary']['finite'];count['selection']+=1
  ids=np.asarray(b['frame_ids']);selected=set(e['planned_positions'])
  while len(selected)<8:
   remaining=[i for i in range(len(ids))if i not in selected];dist={i:min(abs(int(ids[i])-int(ids[j]))for j in selected)for i in remaining};mx=max(dist.values());selected.add(min(i for i in remaining if dist[i]==mx))
  assert sorted(selected)==r['positions8']==b['positions8'];count['position_rules']+=1
  for key,v in s['expert']['observations'].items():assert n['rgb_sha256'][r['positions8'].index(key[1])]==v['receipt']['rgb_sha256'];count['old_RGB_hashes']+=1
  boxes=n['boxes'].numpy().astype(float);pos=e['planned_positions'];take=[r['positions8'].index(i)for i in pos];source=s['Source'].double().numpy();setting=cal['source_domains']['vidstg'if ds=='hc2'else'hc2'];res=encode(boxes[take])-encode(source)[pos];est=estimate([v[None]for v in res],[np.ones(1)for v in res],setting['scale'],setting['truncation'],1/16);re=apply(source,est['delta']);assert np.array_equal(re,b['outputs']['Ref_E_MBR'])and est['selected']==b['estimate']['selected'];count['E_reconstruction']+=1
  # Independently interpolate cxcywh. The conversion to/from xyxy is linear.
  direct=np.stack([np.interp(ids,ids[r['positions8']],boxes[:,k])for k in range(4)],1);err=float(np.max(abs(direct-b['outputs']['Ref_direct8_MBR'])));maximum['geometry']=max(maximum['geometry'],err);assert err<1e-14;count['interpolation']+=1;assert np.array_equal(b['outputs']['E_MBR'],e['E'])and b['interval']==e['interval']
  for arm,bs in b['outputs'].items():
   metric=DenseTube(bs,row,dense[q],spans[q],clip=ds=='hc2').score(e['interval'])
   for ep in ['v','s','t']:
    er=abs(metric[ep]-pub['values'][arm][ep]);assert er<2e-10;maximum['metric']=max(maximum['metric'],er);count['metrics']+=1
   for ep,observed in [('observed_actual_s',True),('unobserved_actual_s',False)]:
    vals=[iou_gt(bs[i],dense[q][int(fid)],row,ds)for i,fid in enumerate(ids)if int(fid)in dense[q]and((i in pos)==observed)];actual=float(np.mean(vals))if vals else None;value=pub['values'][arm][ep];assert(actual is None and value is None)or abs(actual-value)<1e-12;count['actual_nodes']+=1
  for f in [x for x in ff if x['alias']==a]:
   fid=int(ids[f['position']]);assert f['GT_known']==(fid in dense[q])
   if f['GT_known']:
    j=r['positions8'].index(f['position']);er=abs(iou_gt(boxes[j],dense[q][fid],row,ds)-f['teacher_IoU']);assert er<1e-12;count['teacher_anchor']+=1
 for name,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==('hc2'if name=='HC16'else'vidstg')]if name!='Formerly_skipped5'else[r for r in z['rows']if r['formerly_skipped']]
  for pair,dd in g['paired'].items():
   arm=pair.removesuffix('_minus_E')
   for ep,st in dd.items():
    v=np.array([100*(r['values'][arm][ep]-r['values']['E_MBR'][ep])for r in rr if r['values'][arm][ep]is not None and r['values']['E_MBR'][ep]is not None]);ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));assert abs(float(v.mean())-st['mean'])<1e-12 and np.allclose(np.quantile(v[ix].mean(1),[.025,.975]),st['CI95'],rtol=0,atol=1e-12);count['bootstraps']+=1
 ratio=sum(r['Refer_same_sealed_expert_seconds']for r in gate['rows'])/sum(r['T_expert_seconds_without_audit_hashing']for r in gate['rows']);assert ratio==gate['ratio']and ratio<=1.5;verify(True)
 put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=count,max_errors=maximum,CPU_seconds=time.perf_counter()-start,cost_gate_corrected=True,GT_only_postseal=True,no_new_model_inference=True,native_selected_scores_not_full_decoder_GPU_replay=True,original_external_code_unchanged=True));print(count,maximum,flush=True)
if __name__=='__main__':main()
