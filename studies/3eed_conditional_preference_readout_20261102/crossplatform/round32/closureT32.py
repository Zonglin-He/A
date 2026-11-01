"""Actual source and adapted full grounding forwards plus every locked readout."""
import sys,time,random,numpy as np
from common32 import *
from predictT32 import build_selected_libraries,old_memory_library,readouts
import scope32 as readout
sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from closure24 import fp_except
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores

def main():
 import torch
 from transformers import RobertaTokenizerFast
 verify(BASE/'selectedT128_attempt001');out=BASE/'closureT128_attempt001';out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;tokenizer=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');start=time.perf_counter();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));original=fingerprint(model);outside=fp_except(model,'contrastive_align_projection_image.');im={k:v.detach().clone() for k,v in model.contrastive_align_projection_image.state_dict().items()};receipts=[]
 for pl in ['drone','quad']:
  adapted=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));A=[r for r in rows() if r['platform']==pl];libs=build_selected_libraries(A,tokenizer,feedback);oldlib=old_memory_library(A,feedback);n=0;exact=0;maxp=0.;maxbox=0.
  for r in [r for r in rows('T128') if r['platform']==pl]:
   z=feature(r);pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias')});text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');model.contrastive_align_projection_image.load_state_dict(im)
   with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});p0=text_bbf_scores(o,mask)[0].cpu().numpy();bb=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy();query=o['last_proj_queries'][0].cpu().numpy();tokens=o['proj_tokens'][0].cpu().numpy()
   assert np.array_equal(bb,z['boxes']);assert int(p0.argmax())==int(z['scores'].argmax());maxp=max(maxp,float(abs(p0-z['scores']).max()));actual={'scores':p0,'boxes':bb,'query':query,'tokens':tokens,'token_mask':np.asarray(mask,bool)};model.contrastive_align_projection_image.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in adapted.items()})
   with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});pT=text_bbf_scores(o,mask)[0].cpu().numpy();bT=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy()
   be=float(abs(bT-bb).max());assert be==0.;maxbox=max(maxbox,be);cached=load_npz(BASE/'selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));maxp=max(maxp,float(abs(pT-cached['adapted_probability']).max()));ss,_=readouts(r,actual,pT,libs,tokenizer,oldlib);names=list(ss);q=np.array([readout.group_readout(r,s) for s in ss.values()]);assert names==cached['arm_names'].tolist();assert np.array_equal(q,cached['queries']),(r['alias'],[(name,int(a),int(b)) for name,a,b in zip(names,q,cached['queries']) if a!=b]);exact+=len(q);n+=1
  assert fp_except(model,'contrastive_align_projection_image.')==outside;model.contrastive_align_projection_image.load_state_dict(im);assert fingerprint(model)==original;receipts.append({'platform':pl,'observations':n,'full_model_forwards':2*n,'all_readout_argmax_exact':exact,'source_R30_score_maxerror':maxp,'box_maxerror':maxbox,'nonmatching_weights_unchanged':True,'source_restored_bitexact':True});print('CLOSURE32',pl,n,exact,maxp,flush=True)
 dump(out/'CLOSURE_AUDIT.json',receipts);seal(out,'R32_actual_full_source_and_adapted_model_all_readouts_closure',observations=128,forwards=256,seconds=time.perf_counter()-start);print('CLOSURE_T12832_SEALED')
if __name__=='__main__':main()
