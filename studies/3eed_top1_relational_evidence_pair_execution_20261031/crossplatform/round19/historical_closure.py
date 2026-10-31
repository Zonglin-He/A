"""One real full-grounder closure of the historical pair capability diagnostic."""
import sys
import numpy as np
from common import *
from features_v2 import feature
from fit_pair_v2 import optimize
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
def main():
 import torch
 locked('HISTORICAL_CLOSURE_LOCK.json');out=ROOT/'crossplatform/round19/historical_closure_attempt001';out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);layer=model.contrastive_align_projection_image[-1];base={k:v.detach().clone() for k,v in layer.state_dict().items()};results=[]
 for r in read(EXP.parent/'input/A64_PRIVATE.json'):
  z=feature(r);p=read(ROOT/'crossplatform/round17/G_selected64_attempt001/cases_PRIVATE'/(r['id']+'.json'))['preference'];qa=int(z['scores'].argmax())
  if p['status']!='usable' or qa not in p['negative_queries']:continue
  qb=max(p['positive_queries'],key=lambda q:(float(z['scores'][q]),-q));w,before,after,d=optimize(z,qa,qb,read(EXP/'FIT_LOCK_v2.json'),base,'cuda');layer.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in w.items()});row={k:v for k,v in r.items() if k not in ('stage','alias')};pc,_,_=sample_points(row);text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens')
  with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
  from regression_adapter import text_bbf_scores
  boxes=torch.cat([o['last_center'],o['last_pred_size']],-1).cpu().numpy()[0];scores=text_bbf_scores(o,mask).cpu().numpy()[0];assert np.array_equal(boxes,z['boxes']) and int(scores.argmax())==qb and np.allclose(scores,after,rtol=2e-4,atol=2e-6);results.append({'alias':r['alias'],'native_full_model_selects_G_candidate':True,'all256boxes_bitexact':True,'projection_parameters':18496,'steps':d['steps'],'full_vs_cached_max_abs':float(abs(scores-after).max()),'not_newG_success':True});np.savez_compressed(out/(r['id']+'.npz'),scores=scores,selected_box=boxes[qb]);layer.load_state_dict(base)
 assert fingerprint(model)==fp;dump(out/'RESULTS.json',results);seal(out,'R19_historical_pair_actual_full_grounder_closure_not_primary',n=len(results),source_restored=True);print('HISTORICAL_FULL_MODEL_CLOSURE_COMPLETE')
if __name__=='__main__':main()
