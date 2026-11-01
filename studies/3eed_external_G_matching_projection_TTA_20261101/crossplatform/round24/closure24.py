"""Install selected adapted heads in the complete grounding model, run M64.
Checks cached replay and frozen boxes/unchanged non-matching parameters.
"""
import sys,time,hashlib,random,numpy as np
from common24 import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores
from fit24 import module,batch,scores,interact

def fp_except(model,prefix):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():
  if n.startswith(prefix):continue
  h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 select=read(EXP/'LAYER_SELECTED_LOCK.json');rr=rows('M64');out=BASE/'closureM64_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();start=time.perf_counter();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));original=fingerprint(model);im={k:v.detach().clone() for k,v in model.contrastive_align_projection_image.state_dict().items()};tx={k:v.detach().clone() for k,v in model.contrastive_align_projection_text.state_dict().items()};pred={r['id']:{} for r in rr};traces=[]
 for config in select['per_layer']:
  layer=config['layer'];name=config['name'];prefix='contrastive_align_projection_text.' if layer=='text_mlp' else 'contrastive_align_projection_image.';outside=fp_except(model,prefix);times=[];maxerr=0.;maxbox=0.;changes=0
  for platform in ['drone','quad']:
   state=load_npz(BASE/'screen_attempt001/states_PRIVATE'/(platform+'_'+name+'.npz'));m=module(layer);m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()});m.eval().requires_grad_(False)
   if layer=='query_last':model.contrastive_align_projection_image[-1].load_state_dict(m.state_dict())
   elif layer=='query_mlp':model.contrastive_align_projection_image.load_state_dict(m.state_dict())
   elif layer=='text_mlp':model.contrastive_align_projection_text.load_state_dict(m.state_dict())
   for r in [r for r in rr if r['platform']==platform]:
    z=feature(r);pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias')});text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');torch.cuda.synchronize();t=time.perf_counter()
    with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});ss=text_bbf_scores(o,mask)[0];bb=torch.cat([o['last_center'],o['last_pred_size']],-1)[0]
    if layer=='rank_adapter':
     token=o['proj_tokens'][0][torch.tensor(mask,device='cuda')].mean(0);token=token/token.norm().clamp_min(1e-12)
     with torch.inference_mode():ss=ss+m(o['last_proj_queries'][0]*token).squeeze(-1)
    torch.cuda.synchronize();times.append(time.perf_counter()-t);ss=ss.cpu().numpy();bb=bb.cpu().numpy();be=float(abs(bb-z['boxes']).max());assert be==0.,(r['alias'],be);maxbox=max(maxbox,be)
    # A separate cached-feature path must reproduce the full model's readout.
    b=batch([r])
    with torch.no_grad():expected=scores(m,layer,b)[0].cpu().numpy()
    er=float(abs(ss-expected).max());maxerr=max(maxerr,er);assert er<2e-5,(r['alias'],layer,er);qt=int(ss.argmax());assert qt==int(expected.argmax()),(r['alias'],layer,'argmax mismatch');changes+=qt!=int(z['scores'].argmax());pred[r['id']][layer]={'q':qt,'scores':ss}
  assert fp_except(model,prefix)==outside;model.contrastive_align_projection_image.load_state_dict(im);model.contrastive_align_projection_text.load_state_dict(tx);assert fingerprint(model)==original;traces.append({'layer':layer,'name':name,'full_model_forwards':64,'cached_score_max_error':maxerr,'box_max_error':maxbox,'source_query_changes':changes,'nonmatching_state_unchanged':True,'source_restored_bitexact':True,'forward_seconds':float(sum(times))});print('CLOSURE24',layer,changes,maxerr,flush=True)
 names=['source']+[c['layer'] for c in select['per_layer']]
 for r in rr:
  z=feature(r);ix=np.array([int(z['scores'].argmax())]+[pred[r['id']][n]['q'] for n in names[1:]]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),query_indices=ix,boxes=z['boxes'][ix],**{n+'_scores':pred[r['id']][n]['scores'] for n in names[1:]})
 dump(out/'CLOSURE_AUDIT.json',traces);seal(out,'R24_M64_full_model_head_installation_predictions_before_M_labels',n=64,full_model_forwards=256,boxes_bitexact=True,source_restored=True,M_feedback_used_to_fit=False,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),selection_sha256=sha(EXP/'LAYER_SELECTED_LOCK.json'));print('CLOSURE24_SEALED',flush=True)
if __name__=='__main__':main()
