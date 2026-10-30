"""GPU source/cache builder: inference manifest only; never reads annotation containers or scores."""
import os,sys,pathlib,json,hashlib,time,random,argparse,copy,traceback
sys.dont_write_bytecode=True
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torchvision import transforms
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent

def digest(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def normalize(raw,mode):
 if mode=='legacy32':
  x=raw.astype(np.float32);x=x-x.mean(0);x=x/np.linalg.norm(x,axis=1).max()
 else:
  x=raw.copy();x=x-np.mean(x,axis=0);x=x/np.max(np.sqrt(np.sum(x**2,axis=1)))
 return x.astype(np.float32)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['dev','confirm'],required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--gpu-owner',required=True);p.add_argument('--diagnostics',action='store_true');p.add_argument('--engineering-first-parent',action='store_true');a=p.parse_args()
 assert os.environ.get('ROUND2_GPU_OWNER')==a.gpu_owner,'Root-assigned GPU owner environment is required; launch under the unified flock'
 execution_lock=ROOT/'EXECUTION_LOCK.json';execution=json.loads(execution_lock.read_text())
 for name,h in execution['source_sha256'].items():assert digest(ROOT/name)==h,('Locked source changed',name)
 for name,h in execution['input_sha256'].items():assert digest(pathlib.Path(name))==h,('Locked input changed',name)
 assert a.cohort=='dev' or not a.diagnostics,'Confirm cannot introduce exploratory operator diagnostics'
 if a.cohort=='confirm':assert (ROOT/'DEV_DECISION_FROZEN.json').exists(),'Freeze development decision before confirmation source execution'
 manifest=ROOT/f'cohorts/{a.cohort}_inference.json';co=json.loads((ROOT/'cohorts/COHORT_LOCK.json').read_text());assert digest(manifest)==co['manifests'][manifest.name];rows=json.loads(manifest.read_text())
 if a.engineering_first_parent:
  assert a.cohort=='dev' and a.diagnostics,'Engineering subset only permits first locked development parent with diagnostics'
  rows=rows[:1]
 a.output.mkdir(parents=True,exist_ok=False)
 status={'state':'running','cohort':a.cohort,'manifest_sha256':digest(manifest),'gpu_owner':a.gpu_owner,'model_outputs_not_scored':True,'normalization':'native rawdtype-center/maxradius thenfloat32','K':5,'diagnostics':a.diagnostics,'engineering_subset_first_locked_parent':a.engineering_first_parent,'execution_lock_sha256':digest(execution_lock)};(a.output/'CACHE_STATUS.json').write_text(json.dumps(status,indent=2))
 os.environ['GREAT_TEXT_MODEL']=str(BASE/'assets/robbert-v2-dutch-base');sys.path.insert(0,str(ROOT/'frozen_base'));from model.GREAT import GREAT
 import model.pointnet2_utils as pu
 model=GREAT(pre_train=False);ckpt=BASE/'downloads/checkpoints/best_unseen_aff.model_only_crc_recovered.pt';assert digest(ckpt)=='2226bc80bc7c369fd2e55852a6ccd5e9baf4310403f971d044f7c8bff9bd17bf'
 sd=torch.load(ckpt,map_location='cpu',weights_only=True)['model'];model.load_state_dict({k.removeprefix('module.'):v for k,v in sd.items()},strict=True);del sd
 model.requires_grad_(False).eval().cuda();before={n:x.detach().cpu().clone() for n,x in model.named_buffers()};tap={'aux_on':False};trace=[]
 def pre_head(m,inputs):tap['hidden']=inputs[0].detach().clone()
 def post_head(m,inputs,out):tap['logits']=out.detach().clone()
 def point_hook(m,inputs,out):
  if tap['aux_on']:tap['hierarchy']=[[x.detach().clone(),y.detach().clone()] for x,y in out]
 def image_hook(m,inputs,out):
  if tap['aux_on']:tap['image_fusion']=out.detach().clone()
 def text_hook(m,inputs,out):
  if tap['aux_on']:tap['object_fusion']=out[1].detach().clone()
 hooks=[model.decoder.out_head[-1].register_forward_pre_hook(pre_head),model.decoder.out_head[-1].register_forward_hook(post_head),model.point_encoder.register_forward_hook(point_hook),model.img_text_fusion.register_forward_hook(image_hook),model.affordance_dictionary_fusion.register_forward_hook(text_hook)]
 original_fps=pu.farthest_point_sample
 def fps(xyz,npoint):
  idx=original_fps(xyz,npoint);trace.append({'npoint':npoint,'first_index':int(idx[0,0].item()),'indices_sha256':hashlib.sha256(idx.detach().cpu().numpy().tobytes()).hexdigest()});return idx
 pu.farthest_point_sample=fps
 normimage=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);row_receipts=[]
 try:
  for i,row in enumerate(rows):
   raw=np.load(row['geometry'],allow_pickle=False);assert raw.shape==(2048,3) and raw.dtype==np.float64
   xyz=normalize(raw,'native64');_,ids=cKDTree(xyz).query(xyz,k=17,workers=1);neighbors=np.stack([z[z!=j][:16] for j,z in enumerate(ids)]);assert neighbors.shape==(2048,16)
   img=normimage(Image.open(row['image']).convert('RGB').resize((224,224))).unsqueeze(0).cuda();point=torch.from_numpy(xyz.T.copy()).unsqueeze(0).cuda();probs=[];logits=[];hidden=[];traces=[];calls=[];operator={}
   for v in range(5):
    s=(row['seed']+104729*v)%(2**31);seed(s);tap['aux_on']=bool(a.diagnostics and v==0);trace.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
    with torch.no_grad():out=model(img,point,[row['human']],[row['object']])
    torch.cuda.synchronize();elapsed=time.perf_counter()-t;assert torch.isfinite(out).all() and torch.isfinite(tap['hidden']).all() and torch.isfinite(tap['logits']).all();assert torch.equal(torch.sigmoid(tap['logits']),out),'Raw head replay differs from native sigmoid'
    probs.append(out[0].cpu().numpy());logits.append(tap['logits'][0].cpu().numpy());hidden.append(tap['hidden'][0].cpu().numpy());traces.append(copy.deepcopy(trace));calls.append({'view':v,'seed':s,'seconds':elapsed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    if a.diagnostics and v==0:
     aux={k:tap[k] for k in ['hierarchy','image_fusion','object_fusion']};source=out.detach().clone();tap['aux_on']=False
     def replay(perm=None,paired_projection=False):
      hierarchy=[[x.detach().clone(),y.detach().clone()] for x,y in aux['hierarchy']]
      if perm is not None:hierarchy[0]=[z[:,:,perm] for z in hierarchy[0]]
      h=model.decoder.reshape.register_forward_hook(lambda m,ins,y:y[:,:,perm]) if paired_projection else None
      try:
       with torch.no_grad():q=model.decoder(aux['object_fusion'].clone(),aux['image_fusion'].clone().permute(0,2,1),hierarchy)
      finally:
       if h is not None:h.remove()
      return q if perm is None else q[:,perm.argsort(),:]
     replay_native=replay();assert torch.equal(source,replay_native),'Cached native decoder replay differs; stop before mechanism inference'
     generator=torch.Generator(device='cpu').manual_seed(row['seed']+999);perm=torch.randperm(2048,generator=generator).cuda();index_only=replay(perm);paired=replay(perm,True)
     operator={'cached_native_replay_bitexact':True,'index_only_mean_abs_diff':float((source-index_only).abs().mean()),'index_only_max_abs_diff':float((source-index_only).abs().max()),'paired_projection_restore_max_abs_diff':float((source-paired).abs().max()),'paired_projection_restore_within_1e_5':bool(torch.max((source-paired).abs())<=1e-5)}
     assert operator['paired_projection_restore_within_1e_5'],'Paired projection restoration failed; do not attribute index effect'
   if a.diagnostics:
    seed(row['seed']);tap['aux_on']=False
    with torch.no_grad():repeat_p=model(img,point,[row['human']],[row['object']])[0].cpu().numpy()
    assert np.array_equal(repeat_p,probs[0]),'Same-seed fresh source repeat differs'
    operator['same_seed_fresh_source_repeat_bitexact']=True
   bridge={};payload={'view_probs':np.stack(probs),'view_logits':np.stack(logits),'view_hidden':np.stack(hidden),'neighbors':neighbors.astype(np.int64),'xyz':xyz}
   if a.cohort=='dev':
    seed(row['seed']);tap['aux_on']=False;legacy=torch.from_numpy(normalize(raw,'legacy32').T.copy()).unsqueeze(0).cuda()
    with torch.no_grad():legacy_p=model(img,legacy,[row['human']],[row['object']])[0].cpu().numpy()
    payload['legacy32_source']=legacy_p;bridge={'source_output_max_abs_diff':float(np.abs(legacy_p-probs[0]).max()),'source_output_mean_abs_diff':float(np.abs(legacy_p-probs[0]).mean()),'legacy_bridge_is_preprocessing_diagnostic_not_TTA':True}
   np.savez_compressed(a.output/(row['id']+'.npz'),**payload);receipt={'id':row['id'],'parent_id':row['parent_id'],'native_index':row['native_index'],'view_calls':calls,'fps_traces':traces,'unique_FPS_trace_hashes':len({json.dumps(z,sort_keys=True) for z in traces}),'operator_diagnostics':operator,'normalization_bridge':bridge,'label_data_read':False};(a.output/(row['id']+'.json')).write_text(json.dumps(receipt,indent=2));row_receipts.append(receipt);print(i,row['id'],'source_calls',5+(a.cohort=='dev')+a.diagnostics,'bridge',bridge,flush=True)
   assert all(torch.equal(x.detach().cpu(),before[n]) for n,x in model.named_buffers()),'Frozen source buffer changed'
  seal={p.name:digest(p) for p in sorted(a.output.glob('*.npz'))};(a.output/'CACHE_SEALED.json').write_text(json.dumps(seal,indent=2));status.update(state='sealed',count=len(rows),all_source_buffers_unchanged=True,real_primary_source_calls=len(rows)*5,extra_legacy_bridge_source_calls=len(rows) if a.cohort=='dev' else 0,extra_native_repeat_source_calls=len(rows) if a.diagnostics else 0);(a.output/'CACHE_STATUS.json').write_text(json.dumps(status,indent=2))
 except Exception as e:
  status.update(state='invalid',error=repr(e),traceback=traceback.format_exc());(a.output/'CACHE_STATUS.json').write_text(json.dumps(status,indent=2));raise
 finally:
  pu.farthest_point_sample=original_fps
  for h in hooks:h.remove()
if __name__=='__main__':main()
