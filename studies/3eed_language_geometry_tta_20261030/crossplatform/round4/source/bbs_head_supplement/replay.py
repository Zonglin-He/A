"""CPU-only frozen official final-head replay on sealed query features. No labels."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
import torch
R=Path(__file__).resolve().parent;S=R.parent;P=S.parents[1];sys.path[:0]=[str(P/'upstream')]
from models.modules import ClsAgnosticPredictHead

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_head():
 ck=P/'assets/final_6384/ckpt_6384.pth';assert sha(ck)=='bfdedb97971fb31640c5a78790716dc9547953caa646489138f048375e645ec8'
 with torch.serialization.safe_globals([argparse.Namespace]):full=torch.load(ck,map_location='cpu',weights_only=True)['model']
 prefix='prediction_heads.5.';state={k.removeprefix('module.')[len(prefix):]:v for k,v in full.items() if k.removeprefix('module.').startswith(prefix)};del full
 model=ClsAgnosticPredictHead(256,1,256,seed_feat_dim=288,objectness=False,heading=False,compute_sem_scores=True);model.load_state_dict(state,strict=True);model.requires_grad_(False).eval();assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());return model,state

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();pp=R/'PLAN.json';plan=json.loads(pp.read_text());assert sha(pp)==(R/'PLAN.sha256').read_text().strip()
 for f,h in plan['code_sha256'].items():assert sha(R/f)==h
 source=S/(a.cohort+'_source');seal=json.loads((source/'sealed.json').read_text());assert sha(source/'sealed.json')==plan['source_seal_sha256'][a.cohort]
 out=R/(a.cohort+'_replayed');out.mkdir(exist_ok=False);torch.set_num_threads(4);head,state=load_head();buffers={k:v.clone() for k,v in head.named_buffers()};versions={k:v._version for k,v in head.named_parameters()};traces=[]
 for name,h in sorted(seal['files'].items()):
  start=time.perf_counter();assert sha(source/name)==h
  with np.load(source/name,allow_pickle=False) as z:features=z['query_features'].copy();base=z['query_points_xyz'].copy();boxes=z['boxes'].copy();mask=z['noun_mask'].copy()
  assert features.shape==(256,288) and base.shape==(256,3);mask256=np.zeros(256,bool);assert len(mask)<=256;mask256[:len(mask)]=mask
  with torch.inference_mode():
   data={};center,size=head(torch.from_numpy(features.T.copy())[None],torch.from_numpy(base)[None],data,prefix='last_');logits=data['last_sem_cls_scores'][0];prob=logits.softmax(-1);score=(prob*torch.from_numpy(mask256)).sum(-1)
  center=center[0].numpy();size=size[0].numpy();errc=float(np.max(np.abs(center-boxes[:,:3])));errs=float(np.max(np.abs(size-boxes[:,3:6])));assert max(errc,errs)<=plan['CPU_box_bridge_max_abs_tolerance_m'],(name,errc,errs)
  arrays={'bbs_logits':logits.numpy(),'bbs_probabilities':prob.numpy(),'bbs_semantic_scores':score.numpy(),'bbs_rank_stable':np.argsort(-score.numpy(),kind='stable'),'noun_mask256':mask256,'replayed_center':center,'replayed_size':size};assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(out/name,**arrays)
  traces.append({'name':name,'center_bridge_max_abs_m':errc,'size_bridge_max_abs_m':errs,'top_query':int(arrays['bbs_rank_stable'][0]),'seconds':time.perf_counter()-start});print(json.dumps(traces[-1]),flush=True)
 assert versions=={k:v._version for k,v in head.named_parameters()} and all(torch.equal(v,dict(head.named_buffers())[k]) for k,v in buffers.items())
 (out/'trace.json').write_text(json.dumps(traces,indent=2)+'\n');receipt={'cohort':a.cohort,'source_seal_sha256':sha(source/'sealed.json'),'plan_sha256':sha(pp),'head_state_keys':sorted(state),'strict_state_entry_count':len(state),'head_parameter_count':sum(p.numel() for p in head.parameters()),'all_head_parameters_frozen':True,'buffers_bitexact':True,'CPU_only':True,'no_GT_access':True,'source_boxes_and_PCA_unchanged':True,'source_box_bridge_center_max_abs_m':max(x['center_bridge_max_abs_m'] for x in traces),'source_box_bridge_size_max_abs_m':max(x['size_bridge_max_abs_m'] for x in traces),'files':{p.name:sha(p) for p in sorted(out.glob('*.npz'))}}
 (out/'sealed.json').write_text(json.dumps(receipt,indent=2)+'\n');print('BBS_SEALED',len(traces),sha(out/'sealed.json'))
if __name__=='__main__':main()
