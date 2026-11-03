"""Read-only hooks on original native attention; no modification to forward."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,source_capture,bridge_capture,fingerprint,array_sha
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();p=read(a.plan)
 assert p['n']==336 and p['max_main_forwards']==336 and p['no_parameter_updates']
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h,rel
 out=ROOT/p['capture_root'];binding=sha(a.plan)
 if out.exists():assert a.resume and read(out/'RUN_BINDING.json')['plan_sha256']==binding and not (out/'SEALED.json').exists()
 else:out.mkdir(parents=True);dump(out/'RUN_BINDING.json',{'plan_sha256':binding})
 for name in ['samples_PRIVATE','calls_PRIVATE','cases_PRIVATE']:(out/name).mkdir(exist_ok=True)
 import torch
 sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
 torch.manual_seed(0);np.random.seed(0);start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);versions=[(n,x._version) for n,x in model.named_parameters()]
 rows=read(ROOT/p['manifest']);canon=ROOT/p['canonical_root'];cs=read(canon/'SEALED.json');done=[];new=0;native_bad=0;counts={}
 try:
  for row in rows:
   sid=row['id'];target=out/'samples_PRIVATE'/(sid+'.npz');call=out/'calls_PRIVATE'/(sid+'.json')
   if call.exists():
    prior=read(call);assert a.resume and prior['state']=='COMPLETED' and sha(target)==prior['output_sha256'];done.append(sid);native_bad+=not prior['native_fp2_index_xyz_exact'];continue
   points,rawids,nraw=sample_points(row);cache={};handles=[]
   def att_hook(key):
    def hook(m,args,value):
     assert isinstance(value,tuple) and value[1] is not None;cache[key]=value[1].detach().clone()
    return hook
   for i,layer in enumerate(model.cross_encoder.layers):handles.append(layer.cross_layer.cross_lv.register_forward_hook(att_hook('token_point_'+str(i))))
   for i,layer in enumerate(model.decoder):
    handles.append(layer.cross_v.register_forward_hook(att_hook('query_point_'+str(i))));handles.append(layer.cross_l.register_forward_hook(att_hook('query_token_'+str(i))))
    def query_hook(m,args,val,idx=i):cache['decoder_query_'+str(idx)]=val.detach().clone()
    handles.append(layer.register_forward_hook(query_hook))
   def output_hook(m,args,val):cache['out']=val
   handles.append(model.register_forward_hook(output_hook));dump(call,{'state':'STARTED','plan_sha256':binding,'model_forward_entered':False})
   def entered():
    nonlocal new
    assert new<len(rows);new+=1;dump(call,{'state':'STARTED','plan_sha256':binding,'model_forward_entered':True})
   try:arrays,rec=source_capture(model,points,rawids,row['caption'],retain_prefix=True,on_forward_enter=entered)
   finally:
    for h in handles:h.remove()
   z=cache.pop('out');seed=z['fp2_xyz'][0];native=z['fp2_inds'][0].long();first=z['sa_ins'][1][0].long();second=z['sa_ins'][2][0].long();composed=first[second]
   x=torch.as_tensor(points,device=seed.device);correct=torch.equal(x[composed],seed);assert correct,'Composed sampling index must exactly recover seed coordinates'
   native_ok=bool(torch.equal(x[native],seed));native_bad+=not native_ok
   qseed=z['query_points_sample_inds'][0].long();assert torch.equal(seed[qseed],z['query_points_xyz'][0])
   for key in ['fp2_features','seed_features','text_memory','fp2_xyz']:arrays[key]=z[key][0].detach().cpu().numpy()
   arrays.update(native_fp2_input_indices=native.cpu().numpy(),composed_fp2_input_indices=composed.cpu().numpy(),seed_raw_indices=rawids[composed.cpu().numpy()],input_sample_xyz=points)
   for key,value in cache.items():arrays[key]=value[0].detach().cpu().numpy()
   assert arrays['fp2_features'].shape==arrays['seed_features'].shape==(288,1024)
   assert arrays['query_point_5'].shape==(256,1024) and arrays['token_point_2'].shape[1]==1024
   for key in [k for k in arrays if k.startswith(('query_point_','query_token_','token_point_'))]:
    v=arrays[key];assert np.isfinite(v).all() and (v>=0).all() and np.allclose(v.sum(-1),1,atol=3e-6,rtol=0),key
   canonical=canon/'samples_PRIVATE'/(sid+'.npz');assert sha(canonical)==cs['files'][canonical.relative_to(canon).as_posix()]
   with np.load(canonical,allow_pickle=False) as old:bridge=bridge_capture(arrays,old['boxes'],old['scores']);assert np.array_equal(arrays['point_indices'],old['point_indices']) if 'point_indices' in old.files else True
   with target.open('wb') as f:np.savez_compressed(f,**arrays)
   text=row['caption']+' . not mentioned';tokenized=model.tokenizer([text],padding='longest',return_offsets_mapping=True,return_tensors='pt');offsets=tokenized['offset_mapping'][0].numpy()
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'input_sample_sha256':array_sha(points),'indices_sha256':array_sha(rawids),'capture':rec,'source_bridge':bridge,'raw_count':nraw,'token_offsets_PRIVATE':offsets.tolist(),'all_head_averaged_native_weights':True,'native_fp2_index_xyz_exact':native_ok,'composed_fp2_index_xyz_exact':correct,'query_seed_xyz_exact':True,'native_max_coordinate_error':float((x[native]-seed).abs().max().item())})
   dump(call,{'state':'COMPLETED','plan_sha256':binding,'model_forward_entered':True,'output_sha256':sha(target),'native_fp2_index_xyz_exact':native_ok});done.append(sid);cache.clear();del arrays,z;torch.cuda.synchronize()
   if len(done)%16==0:print('LATENT53',len(done),336,'native_bad',native_bad,flush=True)
  assert versions==[(n,x._version) for n,x in model.named_parameters()] and fingerprint(model)==fp
  counts={'n':len(done),'new_forwards_this_process':new,'forwards_all_attempts':sum(read(c)['model_forward_entered'] for c in (out/'calls_PRIVATE').glob('*.json')),'native_fp2_index_bad_cases':native_bad,'all_composed_index_and_query_seed_checks_pass':True,'model_unchanged':True,'source_fingerprint':fp,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'optimizer_steps':0}
  dump(out/'COUNTS.json',counts);dump(out/'SEALED.json',{'state':'R53_FROZEN_NATIVE_LATENT_CAPTURE_NO_GT',**counts,'plan_sha256':binding,'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()}});print(json.dumps(counts),flush=True)
 except BaseException as e:dump(out/('FAILED_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'error':str(e),'completed_n':len(done),'model_forwards_this_process':new,'no_auto_retry':True});raise
if __name__=='__main__':main()
