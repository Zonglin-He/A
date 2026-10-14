"""Root diagnostic oracle/teacher frame-IoU audit after complete grid/model seals."""
import sys,json,math,hashlib,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.stvg_temporal_caption_query_v1 import ROOT,roster,catalog,read,write,sha
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
from scripts.stvg_box_scale_grid_v1 import iou_cxcywh_numpy
B=ROOT/'artifacts/stvg_box_scale_global_evidence_v1'
def alias(cid):return ('HC'if cid.startswith('hc2')else 'V')+f"{int(cid.split('_')[-1])+1:02d}"
def lse(v):return float(np.max(v)+np.log(np.exp(v-np.max(v)).sum()))
def main():
 rt=read(B/'RUNTIME.json');seal=read(B/'TEACHER_GLOBAL_BARRIER.json');grid=read(B/'GRID_INPUT_BARRIER.json');jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];assert seal['status']=='complete'and seal['count']==len(jobs)==160 and seal['GT_read']is False;assert seal['runtime_sha256']==sha(B/'RUNTIME.json')and seal['grid_input_barrier_sha256']==sha(B/'GRID_INPUT_BARRIER.json')and seal['job_lock_sha256']==sha(B/'TEACHER_JOB_LOCK.json')
 for p,h in rt['pins'].items():assert sha(ROOT/p)==h
 for f in grid['files']+seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 data={};maxerror=0.
 for j in jobs:
  z=read(B/'teacher'/f"{j['case_id']}_{j['kind']}_{j['window_id']:03d}.json");assert z['input_file_sha256']==j['input_file_sha256']and z['source_case_sha256']==j['source_case_sha256']and z['prompt_sha256']==j['prompt_sha256']and z['GT_read']is False;a=np.array(z['label_logits']);prob=np.exp(a-lse(a));maxerror=max(maxerror,float(np.max(abs(prob-z['probabilities']))),abs((a[0]-lse(a[1:]))-z['referent_support_log_odds']));data[(j['case_id'],j['kind'],j['window_id'])]=z
 assert maxerror<3e-6
 cases={r['case_id']:catalog(r)[0]for r in roster()};truths,bindings=_truths_after_seal(cases);rows=[];cand=[]
 for r in grid['records']:
  cid=r['case_id'];c=cases[cid];g=truths[cid];fid=r['physical_frame_id'];windows=r['grid']['windows'];original=data[(cid,'original',0)];scores={}
  for kind in ['masked','marked']:
   vals=[original['referent_support_log_odds']-data[(cid,kind,w['window_id'])]['referent_support_log_odds']if kind=='masked'else data[(cid,kind,w['window_id'])]['referent_support_log_odds']for w in windows];scores[kind]=vals
  best={k:int(np.argmax(v))for k,v in scores.items()};truth=g['truth'].get(fid);available=truth is not None
  if available:
   gt=np.array(truth,float);assert gt.shape==(4,);gt=gt/np.array([c['width'],c['height'],c['width'],c['height']]);target=np.r_[(gt[:2]+gt[2:])/2,gt[2:]-gt[:2]];boxes=np.array([[w['cx'],w['cy'],w['w'],w['h']]for w in windows]);ious=iou_cxcywh_numpy(boxes,target);source=float(iou_cxcywh_numpy(r['grid']['student_baseline']['cxcywh'],target));oracle=float(ious.max());ms=float(ious[best['masked']]);mark=float(ious[best['marked']])
  else:ious=[None]*len(windows);source=oracle=ms=mark=None
  row=dict(case=alias(cid),dataset=r['dataset'],window_count=len(windows),GT_anchor_available=available,GT_unknown_reason=None if available else 'prelocked anchor lacks saved GT annotation; no GT-informed reselection',source_IoU=source,oracle_IoU=oracle,mask_selected_IoU=ms,marked_selected_IoU=mark,delta_oracle=None if not available else oracle-source,delta_mask=None if not available else ms-source,delta_marked=None if not available else mark-source,original_support_probability=original['probabilities'][0],original_preferred_label=original['preferred_label'],mask_selected_window_id=windows[best['masked']]['window_id'],marked_selected_window_id=windows[best['marked']]['window_id'],mask_max_response=max(scores['masked']),mask_negative_response_count=sum(v<0for v in scores['masked']),mask_zero_response_count=sum(v==0for v in scores['masked']),marked_selected_label=data[(cid,'marked',windows[best['marked']]['window_id'])]['preferred_label'],mask_top_exact_tie_count=sum(v==max(scores['masked'])for v in scores['masked']),marked_top_exact_tie_count=sum(v==max(scores['marked'])for v in scores['marked']))
  rows.append(row)
  for i,w in enumerate(windows):cand.append(dict(case=alias(cid),dataset=r['dataset'],window_id=w['window_id'],GT_IoU=None if not available else float(ious[i]),mask_response=scores['masked'][i],marked_score=scores['marked'][i],mask_label=data[(cid,'masked',w['window_id'])]['preferred_label'],marked_label=data[(cid,'marked',w['window_id'])]['preferred_label']))
 out=dict(status='actual_P0_postseal_metrics_pending_complete_pixel_source_readback',rows=rows,candidates=cand,count=8,grid_window_count=76,teacher_requests=160,GT_read=True,GT_for_generation_or_selection=False,cohort_historically_exposed=True,score_recomputation_max_error=maxerror,cost=dict(actual_calls=160,total_seconds=sum(z['total_seconds']for z in data.values()),input_tokens=sum(z['input_tokens']for z in data.values()),visual_tokens=sum(z['visual_tokens']for z in data.values()),output_tokens=0,GPU_peak_bytes=max(z['GPU_peak_bytes']for z in data.values()),new_student_fits=0,new_model_downloads=0))
 write(B/'ROOT_PRELIMINARY_POSTSEAL_METRICS.json',out)
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds and r['GT_anchor_available']];print(ds,'valid',len(rr),{k:statistics.mean(r[k]for r in rr)if rr else None for k in ['source_IoU','oracle_IoU','mask_selected_IoU','marked_selected_IoU','delta_oracle','delta_mask','delta_marked']})
 print('score arithmetic max error',maxerror)
if __name__=='__main__':main()
