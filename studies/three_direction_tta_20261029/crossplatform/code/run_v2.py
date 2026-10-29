import argparse,os,sys,json,time,hashlib,types
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'));os.chdir(ROOT/'upstream')
from models.bdetr import BeaUTyDETR
from utils.transform_waymo import transform_to_front_view
from tta_v2 import Adapter,text_span,boxes,scores,match

def load_input(s,seed):
 pc=np.load(s['lidar_path']) if s['lidar_path'].endswith('.npy') else np.fromfile(s['lidar_path'],np.float32).reshape(-1,4)
 rng=np.random.default_rng(seed);idx=rng.choice(len(pc),16384,replace=len(pc)<16384);xyz=pc[idx,:3].copy()
 if s['platform']=='drone':xyz[:,2]+=1.8
 if s['platform']=='waymo':
  view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])]
  xyz,_=transform_to_front_view(xyz,np.zeros((1,7)),view)
 return {'point_clouds':torch.tensor(xyz,dtype=torch.float32,device='cuda')[None],'text':[s['text']]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=99);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((ROOT/'manifest.json').read_text())[:a.limit]
 plan={'version':1,'seed':20261029,'samples':[s['id'] for s in manifest],'steps':[0,1,3],'modes':['entropy','consistency','gated'],'lr':1e-4,'drop':.1,'query_temperature':1.,'query_distribution':'noun scores normalized across query dimension; teacher averages normalized view distributions','gate_iou':.25,'parameters':'cross_encoder LayerNorm affine','protocol':'mixed-checkpoint target-only episodic','labels_used_for_update':False,'readout':'text-only first object lexicon match','checkpoint':str(Path(a.checkpoint).resolve()),'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['tta_v2.py','run_v2.py','prepare.py','score.py']},'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'torch_version':torch.__version__,'registered_plan_sha256':hashlib.sha256((ROOT/'PLAN_LOCK_v6.json').read_bytes()).hexdigest()}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'plan.sha256').write_text(hashlib.sha256((out/'plan.json').read_bytes()).hexdigest())
 torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);torch.set_num_threads(4)
 m=BeaUTyDETR(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,contrastive_align_loss=True,butd=False,self_attend=True,roberta_config_only=True)
 with torch.serialization.safe_globals([argparse.Namespace]):
  ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 state=ck.get('model',ck.get('model_state_dict',ck.get('state_dict',ck)))
 state={k.removeprefix('module.'):v for k,v in state.items()}
 if 'text_encoder.embeddings.word_embeddings.weight' not in state:raise RuntimeError('Checkpoint lacks complete frozen text encoder; no random fallback allowed')
 m.load_state_dict(state,strict=True)
 if not all(torch.isfinite(v).all() for v in m.state_dict().values() if v.is_floating_point()):raise RuntimeError('Nonfinite pretrained state')
 del ck,state
 m.cuda().eval();adapter=Adapter(m)
 records=[]
 def save(s,arm,o):
  token_scores=((o['last_proj_queries']@o['proj_tokens'].transpose(-1,-2))/.07).softmax(-1)[0]
  dest=out/f"{s['id']}_{arm}.npz";np.savez_compressed(dest,boxes=boxes(o).detach().cpu().numpy(),token_scores=token_scores.detach().cpu().numpy())
 for i,s in enumerate(manifest):
  start=time.perf_counter();torch.cuda.reset_peak_memory_stats();inputs=load_input(s,plan['seed']+i);mask,noun=text_span(m.tokenizer,s['text']);traces={}
  for mode in plan['modes']:
   preds,teacher,diag=adapter.run(inputs,mask,3,mode=mode,seed=plan['seed']+i)
   if mode=='entropy':
    save(s,'baseline',preds[0]);save(s,'sparse_no_update',teacher);baseline_boxes=boxes(preds[0]).detach().clone();baseline_scores=scores(preds[0],mask).detach().clone()
    # Same-information no-update ensemble, geometrically match proposals.
    b0,bt=boxes(preds[0]),boxes(teacher);align=match(b0,bt)
    ps=((preds[0]['last_proj_queries']@preds[0]['proj_tokens'].transpose(-1,-2))/.07).softmax(-1)[0]
    pt=((teacher['last_proj_queries']@teacher['proj_tokens'].transpose(-1,-2))/.07).softmax(-1)[0][align]
    np.savez_compressed(out/f"{s['id']}_ensemble_no_update.npz",boxes=b0.detach().cpu().numpy(),token_scores=((ps+pt)/2).cpu().numpy())
   for step in (1,3):save(s,f'{mode}_{step}',preds[step])
   traces[mode]=diag
  adapter.reset()
  with torch.no_grad():source_repeat=m(inputs)
  reset_diff=max(float((boxes(source_repeat)-baseline_boxes).abs().max()),float((scores(source_repeat,mask)-baseline_scores).abs().max()))
  if reset_diff>1e-5:raise RuntimeError(f'Episodic reset/source repeat failed: {reset_diff}')
  torch.cuda.synchronize();rec={'id':s['id'],'platform':s['platform'],'noun':noun,'reset_max_abs_diff':reset_diff,'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated(),'diagnostics':traces};records.append(rec)
  (out/'trace.json').write_text(json.dumps(records,indent=2));print(json.dumps({k:v for k,v in rec.items() if k!='diagnostics'}),flush=True)
 adapter.reset()
 hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in out.glob('*.npz')};(out/'sealed.json').write_text(json.dumps({'files':hashes,'samples':len(manifest),'plan_sha256':hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()},indent=2))
 print('SEALED',len(hashes),flush=True)
if __name__=='__main__':main()
