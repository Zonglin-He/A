"""Actual grounder forwards for accepted episodes; source and all boxes fixed."""
import sys,time
import numpy as np
from common import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
def main():
 import torch
 locked('CLOSURE_LOCK.json');fit=ROOT/'crossplatform/round19/pair_fit96_attempt001';verify(fit);out=ROOT/'crossplatform/round19/closed96_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);layer=model.contrastive_align_projection_image[-1];base={k:v.detach().clone() for k,v in layer.state_dict().items()};trace=[]
 from regression_adapter import text_bbf_scores
 for r in read(EXP.parent/'input/ALL96_PRIVATE.json'):
  z=feature(r);wp=fit/'weights_PRIVATE'/(r['id']+'.npz')
  with np.load(fit/'predictions_PRIVATE'/(r['id']+'.npz')) as a:names=a['arm_names'].copy();indices=a['query_indices'].copy();boxes=a['arm_boxes_raw6'].copy();after=a['after_scores'].copy() if 'after_scores' in a else None
  d={'alias':r['alias'],'id_PRIVATE':r['id'],'accepted':wp.exists(),'boxes_frozen_verified':True}
  if wp.exists():
   with np.load(wp) as a:layer.load_state_dict({k:torch.tensor(a[k],device='cuda') for k in ('weight','bias')})
   row={k:v for k,v in r.items() if k not in ('stage','alias','frame_alias')};pc,_,_=sample_points(row);text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens')
   with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
   bb=torch.cat([o['last_center'],o['last_pred_size']],-1).cpu().numpy()[0];ss=text_bbf_scores(o,mask).cpu().numpy()[0];assert np.array_equal(bb,z['boxes']);err=float(np.max(abs(ss-after)));assert np.allclose(ss,after,rtol=2e-4,atol=2e-6);d.update(score_replay_max_abs=err,cached_top1_equals_full=bool(ss.argmax()==indices[2]));indices[2]=int(ss.argmax());boxes[2]=bb[indices[2]];layer.load_state_dict(base)
  np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=names,query_indices=indices,arm_boxes_raw6=boxes);trace.append(d)
 assert fingerprint(model)==fp;dump(out/'TRACE_PRIVATE.json',trace);seal(out,'R19_full_grounder_closure_sealed_before_current_labels',n=96,actual_updated_grounder_forwards=sum(r['accepted'] for r in trace),all_candidate_boxes_bitexact=True,source_restored=True);print('CLOSURE_COMPLETE',sum(r['accepted'] for r in trace))
if __name__=='__main__':main()
