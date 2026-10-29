import os,sys,json,hashlib,argparse,time
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(P/'upstream')];os.chdir(P/'upstream')
from common import reconstruct_points
from tta import text_span
from models.bdetr import BeaUTyDETR
plan=json.loads((R/'PLAN_LOCK.json').read_text())
assert hashlib.sha256((R/'PLAN_LOCK.json').read_bytes()).hexdigest()==(R/'PLAN_LOCK.sha256').read_text().strip()
for name,h in plan['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,(name,'source mismatch')
assert hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()==plan['manifest_sha256'];manifest=json.loads((R/'manifest.json').read_text());assert len(manifest)==36
out=R/'source';out.mkdir(exist_ok=False);(out/'registered_plan.json').write_bytes((R/'PLAN_LOCK.json').read_bytes());torch.manual_seed(20261029);torch.set_num_threads(4)
m=BeaUTyDETR(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,contrastive_align_loss=True,butd=False,self_attend=True,roberta_config_only=True)
with torch.serialization.safe_globals([argparse.Namespace]):checkpoint=torch.load(P/'assets/final_6384/ckpt_6384.pth',map_location='cpu',weights_only=True)
state={k.removeprefix('module.'):v for k,v in checkpoint['model'].items()};m.load_state_dict(state,strict=True);assert all(torch.isfinite(x).all() for x in state.values() if x.is_floating_point());del checkpoint,state
m.requires_grad_(False).cuda().eval();trace=[]
with torch.inference_mode():
 for i,s in enumerate(manifest):
  torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats();pc=reconstruct_points(s,20261029+i);inputs={'point_clouds':torch.tensor(pc,dtype=torch.float32,device='cuda')[None],'text':[s['text']]};o=m(inputs);boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0];tokens=((o['last_proj_queries']@o['proj_tokens'].transpose(-1,-2))/.07).softmax(-1)[0];mask,noun=text_span(m.tokenizer,s['text']);score=(tokens*mask.to(tokens.device)[None]).sum(-1);top=int(score.argmax());source7=np.r_[boxes[top].cpu().numpy(),np.float32(0)].astype(np.float32);assert torch.isfinite(boxes).all() and torch.isfinite(tokens).all();np.savez_compressed(out/f"{s['id']}.npz",boxes=boxes.cpu().numpy(),token_scores=tokens.cpu().numpy(),source7d=source7,top_query_index=np.int64(top));torch.cuda.synchronize();record={'id':s['id'],'platform':s['platform'],'scene':s['scene'],'seed':20261029+i,'noun':noun,'top_query_index':top,'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated()};trace.append(record);print(json.dumps(record),flush=True)
(out/'trace.json').write_text(json.dumps(trace,indent=2));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')};(out/'sealed.json').write_text(json.dumps({'files':hashes,'n':len(trace),'plan_sha256':hashlib.sha256((R/'PLAN_LOCK.json').read_bytes()).hexdigest(),'label_access':False},indent=2));print('SOURCE SEALED',len(trace),flush=True)
