"""Full source/R30 forward closure with original saved GPU retrieval indices."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,random,argparse,numpy as np
from common34 import *
from keyscope34 import gpu_neighbors
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores

def outside_fp(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():
  if n.startswith('contrastive_align_projection_image.'):continue
  h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 import torch
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['diagnostic','U128'],required=True);a=p.parse_args();check_plan();pred=output(a.split);verify(pred);assert read(EXP/('CPU_AUDIT_'+a.split+'.json'))['state']=='PASSED';out=BASE/('closure_'+a.split+'_attempt001');out.mkdir(exist_ok=False);(out/'observations_PRIVATE').mkdir()
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));initial=fingerprint(model);outside=outside_fp(model);source={k:v.detach().clone() for k,v in model.contrastive_align_projection_image.state_dict().items()};h=[];hook=model.contrastive_align_projection_image.register_forward_pre_hook(lambda m,x:h.append(x[0].detach().clone()));start=time.perf_counter();receipts=[];rr,selection=closure_rows(a.split);dump(out/'CLOSURE_SELECTION.json',selection)
 for pl in ['drone','quad']:
  ss=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));state={k:torch.tensor(v,device='cuda') for k,v in ss.items()};libs={k:load_npz(pred/'libraries_PRIVATE'/(pl+'_'+k+'.npz')) for k in KEYS}
  for k,lib in libs.items():lib['keys_GPU']=torch.tensor(lib['keys'],device='cuda')
  for r in [r for r in rr if r['platform']==pl]:
   cached=load_npz(pred/'predictions_PRIVATE'/(r['id']+'.npz'));z=feature(r);pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias','split')});text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');model.contrastive_align_projection_image.load_state_dict(source);h.clear()
   with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});p0=text_bbf_scores(o,mask)[0].cpu().numpy();bb=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy();actual={'h':h[-1][0].cpu().numpy(),'query':o['last_proj_queries'][0].cpu().numpy(),'tokens':o['proj_tokens'][0].cpu().numpy(),'token_mask':np.asarray(mask,bool)}
   model.contrastive_align_projection_image.load_state_dict(state)
   with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]});pT=text_bbf_scores(o,mask)[0].cpu().numpy();bT=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy()
   knn={k:gpu_neighbors(make_key(actual,k),lib) for k,lib in libs.items()};neighbors_equal=all(np.array_equal(knn[k][v],cached[k+'_'+v]) for k in KEYS for v in ['obs','pos']);readouts,_=all_scores(p0,pT,libs,knn);scores=np.stack([readouts[n] for n in ARMS]);queries=np.array([group_readout(r,s) for s in scores]);boxerr=max(float(abs(bb-z['boxes']).max()),float(abs(bT-bb).max()));perr=max(float(abs(p0-cached['source_probability']).max()),float(abs(pT-cached['R30_probability']).max()));qmatch=np.array_equal(queries,cached['queries']);herr=float(abs(actual['h']-z['h']).max());serr=float(abs(scores-cached['complete_scores']).max())
   np.savez_compressed(out/'observations_PRIVATE'/(r['id']+'.npz'),source_probability=p0,R30_probability=pT,complete_scores=scores,queries=queries,boxes=bb[queries]);receipt={'alias':r['alias'],'platform':pl,'source_adapted_probability_maxerror':perr,'complete_coordinate_score_maxerror':serr,'source_hidden_maxerror':herr,'boxes_maxerror':boxerr,'GPU_neighbors_reproduced':neighbors_equal,'all_readout_queries_reproduced':bool(qmatch)};receipts.append(receipt);dump(out/'RUNNING_RECEIPTS.json',receipts)
   assert boxerr==0 and perr<2e-5 and herr<2e-5 and neighbors_equal and qmatch,receipt
  assert outside_fp(model)==outside;model.contrastive_align_projection_image.load_state_dict(source);assert fingerprint(model)==initial
  print('CLOSURE34',a.split,pl,'seconds',round(time.perf_counter()-start,1),flush=True)
 hook.remove();dump(out/'CLOSURE_AUDIT.json',{'state':'PASSED','rows':receipts,'observations':len(rr),'prediction_observations':528 if a.split=='diagnostic' else 128,'closure_scope':'one fixed input-hash observation per diagnostic split/platform; all U128' ,'full_model_forwards':2*len(rr),'source_restored_bitexact':True,'nonmatching_parameters_unchanged':True,'all_GPU_neighbors_and_final_queries_reproduced':True,'prediction_seal_sha256':sha(pred/'SEALED.json')});seal(out,'R34_actual_full_source_R30_model_h_key_and_all_readouts_closure',n=len(rr),forwards=2*len(rr),**plan_receipt(),seconds=time.perf_counter()-start,source_restored=True,box_maxerror=0.);print('CLOSURE34_SEALED')
if __name__=='__main__':main()
