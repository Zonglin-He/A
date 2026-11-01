import sys,time,random,numpy as np
from common30 import *
sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from closure24 import fp_except
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores

def main():
 import torch
 verify(BASE/'selectedS64_attempt001');torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out=BASE/'closureS64_attempt001';out.mkdir(exist_ok=False);start=time.perf_counter();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));original=fingerprint(model);outside=fp_except(model,'contrastive_align_projection_image.');initial={k:v.detach().clone() for k,v in model.contrastive_align_projection_image.state_dict().items()};receipt=[];sel=read(EXP/'SELECTED_BETA_LOCK.json')
 for arm,name in [('preference_only','b0'),('keep_selected',sel['selected']['name'])]:
  maxerr=0.;maxbox=0.;n=0;exact=0
  for pl in ['drone','quad']:
   state=load_npz(BASE/'screen_attempt001/states_PRIVATE'/(pl+'_'+name+'.npz'));model.contrastive_align_projection_image.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()})
   for r in [r for r in rows('S64') if r['platform']==pl]:
    z=feature(r);pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias')});text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens')
    with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});ss=text_bbf_scores(o,mask)[0].cpu().numpy();bb=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy()
    p=load_npz(BASE/'selectedS64_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));expected=p[arm+'_scores'];er=float(abs(ss-expected).max());be=float(abs(bb-z['boxes']).max());q=int(p['queries'][p['arm_names'].tolist().index(arm)]);assert er<2e-5 and be==0.;assert q==int(ss.argmax()),(r['alias'],arm,q,int(ss.argmax()));maxerr=max(maxerr,er);maxbox=max(maxbox,be);n+=1;exact+=q==int(ss.argmax())
  assert fp_except(model,'contrastive_align_projection_image.')==outside;model.contrastive_align_projection_image.load_state_dict(initial);assert fingerprint(model)==original;receipt.append({'arm':arm,'full_model_forwards':n,'score_max_abs_error':maxerr,'box_max_abs_error':maxbox,'exact_argmax_matches':exact,'nonmatching_weights_unchanged':True,'source_restored_bitexact':True});print('CLOSURE30',arm,n,maxerr,flush=True)
 dump(out/'CLOSURE_AUDIT.json',receipt);seal(out,'R30_selected_and_preference_fullmodel_closure_before_Sscore',forwards=128,seconds=time.perf_counter()-start);print('CLOSURE30_SEALED')
if __name__=='__main__':main()
