"""Deferred full U forwards of the two new heads, sharing sealed source U cache.
Original R30 full closure is covered by R34; do not repeat it or train states.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,argparse,time,random,numpy as np
from u35_helpers import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores
NEW=['U_rows_matched_denominator','partial_output_keep']

def nonmatching(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():
  if not n.startswith('contrastive_align_projection_image.'):
   h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',default=str(DEFAULT_LOCK));a=p.parse_args();check_u_lock(a.lock);verify(U_OUT);assert read(EXP/'U128_CPU_AUDIT.json')['state']=='PASSED';verify(ROOT/'crossplatform/round33/sourceU128_attempt001');out=U_CLOSURE;out.mkdir(exist_ok=False);(out/'observations_PRIVATE').mkdir()
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));original=fingerprint(model);outside=nonmatching(model);source={k:v.detach().clone() for k,v in model.contrastive_align_projection_image.state_dict().items()};expected=source_head()
 for k,v in source.items():assert np.array_equal(v.cpu().numpy(),expected[k])
 hookvals=[];hook=model.contrastive_align_projection_image.register_forward_pre_hook(lambda m,x:hookvals.append(x[0].detach().clone()));rr=U_rows();start=time.perf_counter();receipts=[]
 for pl in ['drone','quad']:
  for name in NEW:
   state=state_for(pl,name);model.contrastive_align_projection_image.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()});idx=ARMS.index(name)
   for r in [r for r in rr if r['platform']==pl]:
    z=U_feature(r);cached=load_npz(U_OUT/'predictions_PRIVATE'/(r['id']+'.npz'));pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias','split')});text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');hookvals.clear()
    with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});prob=text_bbf_scores(o,mask)[0].cpu().numpy();box=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy()
    pe=float(abs(prob-cached['probability_scores'][idx]).max());he=float(abs(hookvals[-1][0].cpu().numpy()-z['h']).max());be=float(abs(box-z['boxes']).max());q=int(prob.argmax());same=q==int(cached['queries'][idx]);rec={'alias':r['alias'],'platform':pl,'configuration':name,'probability_maxerror':pe,'source_hidden_maxerror':he,'box_maxerror':be,'argmax_same':same};receipts.append(rec);np.savez_compressed(out/'observations_PRIVATE'/(r['id']+'_'+name+'.npz'),probability_scores=prob,query=np.array(q),box=box[q]);dump(out/'RUNNING_RECEIPTS.json',receipts);assert pe<2e-5 and he<2e-5 and be==0 and same,rec
   assert nonmatching(model)==outside;model.contrastive_align_projection_image.load_state_dict(source);assert fingerprint(model)==original;print('CLOSURE_U35',pl,name,'seconds',round(time.perf_counter()-start,1),flush=True)
 hook.remove();dump(out/'CLOSURE_AUDIT.json',{'state':'PASSED','n':128,'new_heads':NEW,'full_model_forwards':256,'all_boxes_bitexact':True,'all_argmax_equal':True,'source_restored_bitexact':True,'nonmatching_parameters_unchanged':True,'rows':receipts,'R30_source_fullclosure_reused_from':'crossplatform/round34/closure_U128_attempt001','prediction_seal_sha256':sha(U_OUT/'SEALED.json'),'root_primary_lock_sha256':sha(a.lock)});seal(out,'R35_two_new_heads_actual_full_model_U128_closure',n=128,forwards=256,new_heads=NEW,seconds=time.perf_counter()-start,root_primary_lock_sha256=sha(a.lock));print('CLOSURE_U35_SEALED')
if __name__=='__main__':main()
