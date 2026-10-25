"""Real official updates, exact native/reset checks and paid-input checkpoints."""
from full_common import *
import gc,math,shutil
import numpy as np,torch
from studies.entropy_tta_v2.run import restore_inputs
from stvg.temporal import probabilities,mass_table
from stvg.readout import prediction
from stvg.current import map_interval
class Engine:
 def __init__(self,ds,loader=None,native_forward=None):
  from stvg.runtime import load_source
  from stvg.native import frozen_native_forward
  self.ds=ds;self.loader=loader or load_source;self.native_forward=native_forward or frozen_native_forward;self.model=None;self.initial=None;self.pending={};self.who=identity();self.mods=baseline().official();self.trials=fixed(R/'outputs/paper_followup_v1/tta_selected.json')[ds];self.parser=None;self.start=time.perf_counter();self.actual=dict(new_Source_queries=0,shared_Source_queries=0,optimizer_iterations=0,decoder_calls=0,queries=0,reused_completed_queries=0,new_expert=0)
 def load(self):
  if self.model is None:
   t=time.perf_counter();self.model=self.loader(read(R/'configs/default.json')['datasets'][self.ds]);dec=self.model.ground_decoder.time_decoder;self.initial={k:v.detach().cpu().clone()for k,v in dec.state_dict().items()};atomic(D/f'{self.ds}_model_load.json',dict(worker=self.who,seconds=time.perf_counter()-t,loads=1,shared_with_structural_collector=(self.ds=='vidstg'),nested_not_additive=True))
  return self.model
 def raw_paths(self,key):
  q=D/'active'/self.ds/key;return q,q/'native_context.pt',q/'native_context.json'
 def checkpoint(self,row,c,origin):
  key=row['id'];cap,par=parent(self.ds,row);check_context(c,cap);q,path,meta=self.raw_paths(key);assert not path.exists()and not meta.exists(),'Preserve unbound native input, no blind overwrite';q.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(c,tmp);os.replace(tmp,path);proof=dict(path=str(path),sha256=sha(path),parent=par,input_binding=binding(row,fixed(RUN/'plan.json')['provider_runtime_fingerprint']),origin=origin,GT_read=False,worker=self.who);atomic(meta,proof);self.pending[key]=(c,proof,True);return proof
 def tapped(self,row,model,base,recs,ids,native_seconds):
  if reused(self.ds,row['id'])or contexts(self.ds,row['id']):return
  from stvg.temporal_hidden import cache_temporal
  cap,par=parent(self.ds,row);assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],cap['native_logits']));assert [x['frame_ids']for x in recs]==[x['frame_ids']for x in cap['records']];c=cache_temporal(model,base,recs,ids);self.checkpoint(row,c,dict(kind='shared_original_spatial_Source',Source_forward_seconds=native_seconds,charged_to_original_collector=True));self.actual['shared_Source_queries']+=1
 def acquire(self,row):
  key=row['id']
  if key in self.pending:return self.pending.pop(key)
  if contexts(self.ds,key):c,e=payload_context(self.ds,key);return c,dict(kind='sealed_existing_native_context',**e),False
  q,path,meta=self.raw_paths(key)
  if meta.exists():
   e=read(meta);assert sha(path)==e['sha256']and e['input_binding']==binding(row,fixed(RUN/'plan.json')['provider_runtime_fingerprint']);c=torch.load(path,map_location='cpu',weights_only=False,mmap=True);return c,e,True
  assert not(q/'Source_attempt.json').exists()and not path.exists(),'Unresolved paid Source attempt, preserve; no blind rerun';q.mkdir(parents=True,exist_ok=True);atomic(q/'Source_attempt.json',dict(worker=self.who,before_Source=True,GT_read=False));model=self.load();model.cuda();cap,par=parent(self.ds,row)
  from stvg.decode import decode
  from stvg.query import QuerySubjectParser
  from stvg.temporal_hidden import cache_temporal
  if self.parser is None:self.parser=QuerySubjectParser(R/'checkpoints/stanza')
  tick=time.perf_counter();frames,ids=decode(row['metadata'],self.ds);assert ids==cap['frame_ids'];torch.manual_seed(20260920);np.random.seed(20260920);st=time.perf_counter();batch,recs,base=self.native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses={'subject':self.parser(row['metadata']['caption'])['subject']}));forward=time.perf_counter()-st;assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],cap['native_logits']));assert [x['frame_ids']for x in recs]==[x['frame_ids']for x in cap['records']];c=cache_temporal(model,base,recs,ids);self.checkpoint(row,c,dict(kind='additional_original_Source_for_missing_temporal_inputs',Source_forward_seconds=forward,acquisition_wall_seconds=time.perf_counter()-tick,charged_to_TTA=True));self.actual['new_Source_queries']+=1;assert self.actual['new_Source_queries']<=read(D/'plan.json')['maximum_additional_Source'][self.ds];del batch,base,frames;model.cpu();gc.collect();torch.cuda.empty_cache();return self.pending.pop(key)
 def reuse(self,row):
  key=row['id'];old=reused(self.ds,key);cap,par=parent(self.ds,row);oldroot=PREV
  for f,h in [('path','sha256'),('logits_path','logits_sha256'),('receipt_path','receipt_sha256')]:assert sha(oldroot/old[f])==old[h]
  logs=torch.load(oldroot/old['logits_path'],map_location='cpu',weights_only=False,mmap=True);assert np.array_equal(logs['Source'],cap['Source'])and logs['frame_ids']==cap['frame_ids']
  with np.load(oldroot/old['path'])as z:
   for method in ['Tent','SAR']:
    t=old['traces'][method];assert t['trial']==self.trials[method]and t['reset_exact'];iv=prediction(logs['adapted_logits'][method],torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids'])['physical_interval'];assert np.array_equal(iv,z[method+'Native_interval']);_,q=mass_table(probabilities(cap['records'],logs['adapted_logits'][method]));assert np.array_equal(map_interval(cap['frame_ids'],q),z[method+'JointMAP_interval'])
  out,lp,rp=entry_files(self.ds,key);out.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()and not lp.exists()and not rp.exists();shutil.copyfile(oldroot/old['path'],out);shutil.copyfile(oldroot/old['logits_path'],lp);assert sha(out)==old['sha256']and sha(lp)==old['logits_sha256'];rec=dict(path=str(out.relative_to(D)),sha256=sha(out),logits_path=str(lp.relative_to(D)),logits_sha256=sha(lp),parent=par,reused_completed=True,original=old,traces=old['traces'],worker=self.who,actual_new_counts=dict(new_Source_queries=0,shared_Source_queries=0,optimizer_iterations=0),GT_read=False);atomic(rp,rec);self.actual['reused_completed_queries']+=1
 def process(self,row):
  guard_resources();assert time.perf_counter()-self.start<259200;key=row['id'];out,lp,rp=entry_files(self.ds,key)
  if rp.exists():existing_entry(self.ds,key);return
  assert not out.exists()and not lp.exists();before=dict(self.actual);tick=time.perf_counter()
  if reused(self.ds,key):self.reuse(row)
  else:
   context,origin,owned=self.acquire(row);cap,par=parent(self.ds,row);check_context(context,cap);model=self.load();dec=model.ground_decoder.time_decoder.cuda();head=model.temp_embed.cuda();dec.eval();head.eval();kw=[restore_inputs(x,'cuda')for x in context['offsets']];native=[x['native_logits']for x in context['offsets']]
   with torch.no_grad():assert all(torch.equal(head(dec(**v))[-1].cpu(),n)for v,n in zip(kw,native))
   attempt=D/self.ds/'attempts'/(key+'.json');assert not attempt.exists(),'Preserve previous optimizer attempt, no blind retry';atomic(attempt,dict(before_updates=True,worker=self.who,context_sha256=origin['sha256'],parent=par,GT_read=False));arrays=dict(Source=np.asarray(cap['Source']),Source_interval=np.asarray(prediction(cap['native_logits'],torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids'])['physical_interval']),frame_ids=np.asarray(cap['frame_ids']),positions8=np.asarray(cap['positions8']));alllogs={};traces={};weights={};prior_threads=torch.get_num_threads();torch.set_num_threads(1)
   try:
    for method in ['Tent','SAR']:
     trial=self.trials[method];st=time.perf_counter();dec.load_state_dict(self.initial,strict=True);mod=self.mods[method.lower()];mod.configure_model(dec);dec.eval();head.eval();params,names=mod.collect_params(dec);wrapper=baseline().make_wrapper(dec,head);opt=torch.optim.SGD(params,lr=trial['lr'],momentum=.9)if method=='Tent'else self.mods['sam'].SAM(params,torch.optim.SGD,lr=trial['lr'],momentum=.9);margin=(trial['margin_fraction']or.4)*math.log(max(x.shape[1]for x in native));adapter=mod.Tent(wrapper,opt,steps=1,episodic=False)if method=='Tent'else mod.SAR(wrapper,opt,steps=1,episodic=False,margin_e0=margin,reset_constant_em=.2);steps=[]
     try:
      for step in range(trial['steps']):
       start_trace=len(wrapper.trace);adapter(kw);self.actual['optimizer_iterations']+=1;events=wrapper.trace[start_trace:];steps.append(dict(step=step,forwards=events));assert all(torch.isfinite(x).all()for x in dec.parameters()),'Official nonfinite state; no fallback'
      with torch.no_grad():wrapper(kw);after=[x.detach().cpu().clone()for x in wrapper.last]
      assert all(torch.isfinite(x).all()for x in after);delta=sum(float((v.detach().cpu()-self.initial[k]).square().sum())for k,v in dec.state_dict().items())**.5;arrays[method+'Native']=np.asarray(cap['Source']);arrays[method+'Native_interval']=np.asarray(prediction(after,torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids'])['physical_interval']);_,q=mass_table(probabilities(cap['records'],after));arrays[method+'JointMAP']=np.asarray(cap['Source']);arrays[method+'JointMAP_interval']=np.asarray(map_interval(cap['frame_ids'],q));alllogs[method]=after;weights[method]={name:par.detach().cpu().clone()for name,par in dec.named_parameters()if par.requires_grad};traces[method]=dict(trial=trial,steps=steps,parameter_delta=delta,parameters=names,parameter_scalars=sum(x.numel()for x in params),seconds=time.perf_counter()-st,decoder_calls=wrapper.calls,official_source_unmodified=True,logits_changed=any(not torch.equal(x,y)for x,y in zip(after,native)))
     finally:
      self.actual['decoder_calls']+=wrapper.calls;dec.load_state_dict(self.initial,strict=True);dec.requires_grad_(False);head.requires_grad_(False);dec.eval();head.eval();opt.zero_grad(set_to_none=True)
      for par in dec.parameters():par.grad=None
     with torch.no_grad():reset=[head(dec(**v))[-1].cpu()for v in kw]
     assert all(torch.equal(x,y)for x,y in zip(reset,native))and all(torch.equal(v.detach().cpu(),self.initial[k])for k,v in dec.state_dict().items());traces[method]['reset_exact']=True;del wrapper,adapter,opt,params,after;gc.collect()
   finally:torch.set_num_threads(prior_threads)
   lp.parent.mkdir(parents=True,exist_ok=True);tmp=lp.with_suffix('.tmp');torch.save(dict(Source=np.asarray(cap['Source']),records=cap['records'],frame_ids=cap['frame_ids'],positions8=cap['positions8'],native_logits=native,adapted_logits=alllogs,adapted_affine_weights=weights,GT_read=False),tmp);os.replace(tmp,lp);out.parent.mkdir(parents=True,exist_ok=True)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
   os.replace(out.with_suffix('.tmp'),out)
   with np.load(out)as z:
    cold=torch.load(lp,map_location='cpu',weights_only=False,mmap=True)
    for method,logs in cold['adapted_logits'].items():
     iv=prediction(logs,torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids'])['physical_interval'];_,q=mass_table(probabilities(cap['records'],logs));assert np.array_equal(z[method+'Native_interval'],iv)and np.array_equal(z[method+'JointMAP_interval'],map_interval(cap['frame_ids'],q))
   newcounts={k:self.actual[k]-before[k]for k in ['new_Source_queries','shared_Source_queries','optimizer_iterations']};newcounts['shared_Source_queries']=int(origin.get('origin',{}).get('kind')=='shared_original_spatial_Source');rec=dict(path=str(out.relative_to(D)),sha256=sha(out),logits_path=str(lp.relative_to(D)),logits_sha256=sha(lp),parent=par,context_origin=origin,traces=traces,worker=self.who,reused_completed=False,actual_new_counts=newcounts,query_wall_seconds=time.perf_counter()-tick,all_readouts_cold_exact=True,GT_read=False);atomic(rp,rec)
   if owned:
    q,raw,meta=self.raw_paths(key);assert sha(raw)==read(meta)['sha256'];raw.unlink();atomic(q/'disposal_receipt.json',dict(original_context_sha256=read(meta)['sha256'],adapted_logits_sha256=rec['logits_sha256'],prediction_sha256=rec['sha256'],all_frozen_readouts_replay_exact=True,scope='Only actual acquired input of this new active query removed after committed predictions/logits/receipts. Does not preserve arbitrary future optimizer/context replay.'))
   del cap,context,kw,arrays,alllogs,weights,cold,reset;dec.cpu();head.cpu();gc.collect();torch.cuda.empty_cache()
  self.actual['queries']+=1;atomic(D/f'{self.ds}_progress.json',dict(completed=self.actual['queries'],total=COUNTS[self.ds],actual_counts=self.actual,worker=self.who,GT_read=False));print(self.ds,key,self.actual,flush=True)
 def close(self):
  if self.model is not None:self.model.cpu();self.model=None;gc.collect();torch.cuda.empty_cache()
