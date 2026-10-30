"""Frozen mixed-source forward caches, no adaptation. Explicit parent release required."""
import argparse,hashlib,json,os,sys,time,re
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parent;P=R.parents[1];sys.path[:0]=[str(P),str(P/'upstream')]
from models.bdetr import BeaUTyDETR
from tta import LEXICON
sys.path.insert(0,str(P/'round3/p3'))
from geometry_baselines import crop_source_aabb,orientation_only_from_crop

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(a,b):
 # A one-to-one geometry proposal correspondence, NOT proof of object identity.
 scale=np.maximum(np.median(np.abs(a[:,3:6].astype(float)),axis=0),.5)
 cost=np.linalg.norm((a[:,None,:3]-b[None,:,:3])/scale,axis=-1)+np.linalg.norm((a[:,None,3:6]-b[None,:,3:6])/scale,axis=-1)
 rows,cols=linear_sum_assignment(cost);assert np.array_equal(rows,np.arange(256));return cols.astype(np.int64),cost[rows,cols]
def pca_boxes(boxes,points):
 out=np.c_[boxes,np.zeros(256,dtype=np.float32)].astype(np.float32);accepted=np.zeros(256,bool);count=np.zeros(256,np.int64);reasons=[]
 for j,b in enumerate(boxes):
  crop,d=crop_source_aabb(points,b);ref,diag=orientation_only_from_crop(crop,b,'PCA_standard_axisaware');out[j]=ref;accepted[j]=diag['updated'];count[j]=len(crop);reasons.append(diag['reason'])
 return out,accepted,count,np.array(reasons,dtype='<U48')

def masks(tokenizer,text,caption_length):
 tok=tokenizer(text,return_offsets_mapping=True,return_tensors='pt');off=tok['offset_mapping'][0].numpy();content=(off[:,1]>off[:,0]);cap=content&(off[:,0]<caption_length)
 hit=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',text[:caption_length],re.I)
 noun=content&(off[:,1]>hit.start())&(off[:,0]<hit.end()) if hit else content
 assert cap.any() and noun.any()
 return {'noun_mask':noun,'caption_mask':cap,'context_mask':cap&~noun,'offset_mapping':off,'input_ids':tok['input_ids'][0].numpy(),'attention_mask':tok['attention_mask'][0].numpy()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();pp=R/'SOURCE_PLAN.json';plan=json.loads(pp.read_text());assert sha(pp)==(R/'SOURCE_PLAN.sha256').read_text().strip()
 for name,h in plan['local_code_sha256'].items():assert sha(R/name)==h,name
 for name,h in plan['upstream_code_sha256'].items():assert sha(P/name)==h,name
 release=R/(a.cohort.upper()+'_GPU_RELEASE.json');grant=json.loads(release.read_text());assert grant['source_plan_sha256']==sha(pp) and grant['cohort']==a.cohort and grant['authorized'] is True
 if a.cohort=='confirm':assert grant['final_mechanism_lock_sha256'],'Confirm requires final mechanism freeze'
 manifest=R/'inputs'/(a.cohort+'_manifest.json');assert sha(manifest)==plan['input_manifest_sha256'][a.cohort];samples=json.loads(manifest.read_text());out=R/(a.cohort+'_source');out.mkdir(exist_ok=False);(out/'registered_plan.json').write_bytes(pp.read_bytes())
 torch.set_num_threads(4);torch.manual_seed(plan['seed']);os.chdir(P/'upstream');model=BeaUTyDETR(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,contrastive_align_loss=True,butd=False,self_attend=True,roberta_config_only=True)
 ck=P/'assets/final_6384/ckpt_6384.pth';assert sha(ck)==plan['checkpoint_sha256']
 with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(ck,map_location='cpu',weights_only=True)['model']
 state={k.removeprefix('module.'):v for k,v in state.items()};assert len(state)==1034;assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);del state;model.requires_grad_(False).cuda().eval();cache={};param_versions={n:p._version for n,p in model.named_parameters()};buffer_initial={n:b.clone() for n,b in model.named_buffers()}
 def hook(module,arguments):cache['features']=arguments[0].detach().transpose(1,2).contiguous()
 handle=model.prediction_heads[-1].register_forward_pre_hook(hook);traces=[];first=None
 def infer(sample,view,condition):
  fp=Path(view['point_path']);assert sha(fp)==view['point_sha256'];pc=np.load(fp,allow_pickle=False);text=sample['text'] if condition=='full' else sample['noun_text'];length=sample['caption_character_length'] if condition=='full' else sample['noun_caption_character_length'];mm=masks(model.tokenizer,text,length)
  with torch.inference_mode():
   o=model({'point_clouds':torch.as_tensor(pc,dtype=torch.float32,device='cuda')[None],'text':[text]});boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0];q=o['last_proj_queries'][0];t=o['proj_tokens'][0];logits=q@t.T/.07;prob=logits.softmax(-1);noun=torch.as_tensor(mm['noun_mask'],device='cuda');score=(prob*noun).sum(-1)
   assert boxes.shape==(256,6) and q.shape==(256,64) and t.shape[1]==64 and cache['features'].shape==(1,256,288)
   assert torch.max(torch.abs(q.norm(dim=-1)-1))<1e-5 and torch.max(torch.abs(t.norm(dim=-1)-1))<1e-5
   arrays={'boxes':boxes.cpu().numpy(),'semantic_scores':score.cpu().numpy(),'proj_queries':q.cpu().numpy(),'proj_tokens':t.cpu().numpy(),'exact_logits':logits.cpu().numpy(),'token_scores':prob.cpu().numpy(),'query_features':cache['features'][0].cpu().numpy(),'query_points_xyz':o['query_points_xyz'][0].cpu().numpy(),'query_points_sample_inds':o['query_points_sample_inds'][0].cpu().numpy(),**mm}
  assert all(np.isfinite(x).all() for x in arrays.values());
  if condition=='full' and view['view']==0:
   arrays['boxes_pca'],arrays['pca_accepted'],arrays['pca_crop_count'],arrays['pca_reason']=pca_boxes(arrays['boxes'],pc)
  return arrays
 try:
  for s in samples:
   ref=None
   for view in s['views']:
    full=None
    for condition in ['full','noun']:
     torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats();z=infer(s,view,condition);torch.cuda.synchronize()
     if ref is None:ref=z['boxes'].copy()
     if condition=='full':full=z['boxes'].copy()
     z['aligned_to_full_view0'],z['alignment_to_full_view0_cost']=align(ref,z['boxes']);z['aligned_to_full_sameview'],z['alignment_to_full_sameview_cost']=align(full,z['boxes'])
     name=f"{s['id']}_v{view['view']}_{condition}.npz";np.savez_compressed(out/name,**z)
     rec={'id':s['id'],'platform':s['platform'],'view':view['view'],'condition':condition,'top_query':int(z['semantic_scores'].argmax()),'invalid_size_count':int(np.any(z['boxes'][:,3:6]<=0,axis=1).sum()),'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};traces.append(rec);print(json.dumps(rec),flush=True)
     if first is None:first={k:v.copy() for k,v in z.items() if not k.startswith('align')}
  repeat=infer(samples[0],samples[0]['views'][0],'full');diff={k:float(np.max(np.abs(v.astype(float)-repeat[k].astype(float)))) for k,v in first.items() if v.dtype.kind not in 'US'};assert max(diff.values())<=1e-6,diff
  assert np.array_equal(first['pca_reason'],repeat['pca_reason'])
  assert param_versions=={n:p._version for n,p in model.named_parameters()}
  assert all(torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffer_initial.items())
  (out/'repeat_control.json').write_text(json.dumps({'max_differences':diff,'bitexact':all(v==0 for v in diff.values())},indent=2));(out/'trace.json').write_text(json.dumps(traces,indent=2));(out/'sealed.json').write_text(json.dumps({'n_samples':len(samples),'n_forward_caches':len(traces),'source_plan_sha256':sha(pp),'manifest_sha256':sha(manifest),'files':{p.name:sha(p) for p in sorted(out.glob('*.npz'))},'model_strict_loaded':1034,'numeric_GT_read':False,'source_model_updated':False,'all_parameters_requires_grad_false':all(not p.requires_grad for p in model.parameters()),'parameter_versions_unchanged':True,'all_buffers_bitexact':True,'repeat_control_sha256':sha(out/'repeat_control.json')},indent=2));print('SOURCE_SEALED',len(samples),len(traces),flush=True)
 finally:handle.remove()
if __name__=='__main__':main()
