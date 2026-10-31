"""Full source-model B forwards with only the admitted projection update."""
import sys,time
import numpy as np
from common import ROOT,EXP,read,dump,sha,arrsha,verify,check_plan
def main():
 check_plan();spec=read(EXP/'FIT_SPEC_LOCK.json')
 for rel,h in spec['programs'].items():assert sha(ROOT/rel)==h
 out=ROOT/'crossplatform/round16/B_predictions_attempt001';s=verify(out);gates=s['fit'];passed=[p['platform'] for p in gates if p['gate_passed']]
 receipt={'prediction_seal_sha256':sha(out/'SEALED.json'),'passed':True,'gate_passed_platforms':passed,'new_B_G_or_fitting':False,'all_B_rows_retained':32,'full_model_B_forwards':0,'rows':[]}
 if not passed:
  receipt['scope']='Both supervision prerequisites failed; no actual TTA or additional GPU forward. All32 exact source fallbacks verified against immutable source pool.'
  dump(EXP/'B_CLOSURE.json',receipt);print('CLOSURE_NO_TTA_PREREQUISITE',flush=True);return
 sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
 from run_diagnostic import load_model,fingerprint,mask_for
 from runner_helpers import sample_points
 from regression_adapter import text_bbf_scores
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));start=time.perf_counter()
 original={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 for platform in passed:
  with np.load(out/(platform+'_projection_PRIVATE.npz'),allow_pickle=False) as z:
   model.contrastive_align_projection_image[-1].load_state_dict({k:torch.from_numpy(z[k].copy()) for k in ('weight','bias')})
  rows=[r for r in read(EXP.parent/'input/B32_PRIVATE.json') if r['platform']==platform]
  for i,r in enumerate(rows):
   pc,_,_=sample_points(r);text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens')
   with torch.inference_mode():o=model({'point_clouds':torch.from_numpy(pc)[None].cuda(),'text':[text]});scores=text_bbf_scores(o,mask)[0].cpu().numpy();boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy()
   with np.load(out/'predictions_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:
    assert np.array_equal(boxes,z['source_pool']),'Geometry changed'
    err=float(np.max(np.abs(scores-z['TTA_scores'])));assert err<2e-5
    q=int(scores.argmax());cached=int(z['query_indices'][2]);assert q==cached,'CPU/GPU ranking closure failed; preserve and resolve before scoring'
   receipt['rows'].append({'platform':platform,'alias':f'B_{platform}_{i:03d}','boxes_bitexact':True,'TTA_query_same_as_full_model':True,'max_score_error':err});print('CLOSURE',platform,i+1,flush=True)
  for n,v in model.state_dict().items():
   if n not in ('contrastive_align_projection_image.4.weight','contrastive_align_projection_image.4.bias'):assert torch.equal(v.detach().cpu(),original[n]),n
 receipt.update(full_model_B_forwards=len(receipt['rows']),only_two_allowed_projection_tensors_changed=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());dump(EXP/'B_CLOSURE.json',receipt)
 print('CLOSURE_PASSED',flush=True)
if __name__=='__main__':main()
