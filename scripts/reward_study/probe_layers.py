"""Exactly one HC125 initial Replay, read-only activation hooks, no updates/GT."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,gc,copy
from pathlib import Path
import numpy as np,torch
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();T=ROOT/'outputs/layout_reward_dev';W=T/'workspace';O=T/'research/decoder_audit';sys.path.insert(0,str(W));os.chdir(W)
from stvg.runtime import setup,lease,load_source
from stvg.state import Replay

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,z):p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(z,indent=2)+'\n');os.replace(tmp,p)
setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();raw=[];current={'view':-1,'bbox':0,'layer':None};handles=[];model=rp=None
try:
 fingerprint=read(T/'run.json')['fingerprint'];computed=hashlib.sha256((T/'config.json').read_bytes()+b''.join(p.read_bytes()for p in sorted((W/'stvg').glob('*.py')))+(W/'run_trial.py').read_bytes()).hexdigest();assert computed==fingerprint
 cfg=read(T/'config.json')['datasets']['hc2'];hc_development_rows=[json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()and json.loads(l)['dataset']=='hc2'];key=hc_development_rows[125-1]['id'];p=T/'student_cache'/(key+'.pt');assert sha(p)==read(p.with_suffix('.json'))['sha256'];tp=T/'traces'/(key+'.pt');assert sha(tp)==read(T/'predictions'/(key+'.json'))['trace_sha256'];c=torch.load(p,map_location='cpu',weights_only=False);z=torch.load(tp,map_location='cpu',weights_only=False)
 model=load_source(cfg);model.cpu();gc.collect();torch.cuda.empty_cache();model.ground_decoder.decoder.cuda().eval().requires_grad_(False);dec=model.ground_decoder.decoder
 def rootpre(module,args,kwargs):current['view']+=1;current['bbox']=0;assert current['view']in(0,1);raw.append(dict(kind='root_input',view=current['view'],pred_boxes=kwargs['pred_boxes'].detach().cpu(),query_tgt=kwargs['query_tgt'].detach().cpu()))
 handles.append(dec.register_forward_pre_hook(rootpre,with_kwargs=True))
 def normhooks(name):
  def pre(module,args):raw.append(dict(kind='LayerNorm',name=name,view=current['view'],eps=module.eps,normalized_shape=list(module.normalized_shape),input=args[0].detach().cpu()))
  def post(module,args,out):
   event=next(x for x in reversed(raw)if x['kind']=='LayerNorm'and x['name']==name and x['view']==current['view']);event['output']=out.detach().cpu()
  return pre,post
 for name,module in dec.named_modules():
  if isinstance(module,torch.nn.LayerNorm):
   pre,post=normhooks(name);handles.append(module.register_forward_pre_hook(pre));handles.append(module.register_forward_hook(post))
 def layerpre(name):
  def f(module,args,kwargs):
   current['layer']=name;raw.append(dict(kind='decoder_layer_input',name=name,view=current['view'],tensors={k:v.detach().cpu()for k,v in kwargs.items()if k in ['query_tgt','query_pos','query_sine_embed']and torch.is_tensor(v)}))
  return f
 for name,module in dec.named_modules():
  if name.startswith('layers.')and name.count('.')==1:handles.append(module.register_forward_pre_hook(layerpre(name),with_kwargs=True))
 def bboxhook(module,args,out):
  raw.append(dict(kind='bbox_logits',name=current['layer'],view=current['view'],call=current['bbox'],input=args[0].detach().cpu(),logits=out.detach().cpu()));current['bbox']+=1
 handles.append(dec.bbox_embed.register_forward_hook(bboxhook))
 class ProbeReplay(Replay):
  def values(self):
   vv=super().values();self.last=vv;return vv
 rp=ProbeReplay(model,c,c['Source'],c['E'],c['planned_positions'],lambda:None)
 assert rp.calls==1 and current['view']==1,'Exactly one physical Replay (two native offsets) is allowed'
 actual=rp.last['boxes'].detach().cpu();assert torch.equal(actual,z['LayoutOPD']['path'][0]['boxes']);assert torch.equal(rp.last['native'].cpu(),c['Source']);assert all(not p.requires_grad and p.grad is None for p in rp.decoder.parameters())
 assert not rp.state.detach().count_nonzero()
 # All following calculations are detached CPU summaries of the one forward.
 def summarize(x):
  x=x.double().reshape(x.shape[0],-1);return dict(mean=x.mean(-1).tolist(),variance=x.var(-1,unbiased=False).tolist(),RMS=x.square().mean(-1).sqrt().tolist(),L2=x.norm(dim=-1).tolist(),absmax=x.abs().max(-1).values.tolist())
 records=[]
 for e in raw:
  view=e['view'];n=(len(c['Source'])+1-view)//2;target=27 if view==1 else None
  if e['kind']=='LayerNorm':
   st=summarize(e['input']);var=np.asarray(st['variance']);den=np.sqrt(var+e['eps']);r=dict(kind=e['kind'],name=e['name'],view=view,eps=e['eps'],shape=list(e['input'].shape),input=st,denominator=den.tolist(),output=summarize(e['output']))
   if view==1:
    others=np.delete(var,27);r['node55_comparison']=dict(local_index=27,physical_index=55,variance=float(var[27]),denominator=float(den[27]),variance_over_eps=float(var[27]/e['eps']),same_view_other_variance_median=float(np.median(others)),same_view_other_variance_min=float(others.min()),same_view_other_variance_max=float(others.max()),target_to_other_median=float(var[27]/np.median(others))if np.median(others)>0 else None)
  elif e['kind']=='bbox_logits':
   logits=e['logits'].double().reshape(e['logits'].shape[0],4);pp=logits.sigmoid();r=dict(kind=e['kind'],name=e['name'],view=view,call=e['call'],input=summarize(e['input']),logits=logits.tolist(),box=pp.tolist(),sigmoid_derivative=(pp*(1-pp)).tolist())
   if view==1:r['node55_comparison']=dict(physical_index=55,local_index=27,target_box=pp[27].tolist(),target_logits=logits[27].tolist(),target_sigmoid_derivative=(pp[27]*(1-pp[27])).tolist(),same_view_box_width_height_median=pp[:,2:].median(0).values.tolist(),same_view_box_width_height_min=pp[:,2:].min(0).values.tolist())
  elif e['kind']=='decoder_layer_input':r=dict(kind=e['kind'],name=e['name'],view=view,tensors={k:summarize(v)for k,v in e['tensors'].items()})
  else:
   bb=e['pred_boxes'].double().reshape(e['pred_boxes'].shape[0],4);r=dict(kind=e['kind'],view=view,pred_boxes=bb.tolist(),query_tgt=summarize(e['query_tgt']))
  records.append(r)
 provenance=dict(status='completed_one_initial_replay_no_update',GT_read=False,query=key,public_id='H125',physical_node=55,view=1,local_index=27,original_inference_fingerprint=fingerprint,source_encoder_forwards=0,expert_forwards=0,physical_replay_calls=rp.calls,native_offset_decoder_calls=2,original_path_boxes_bit_exact=True,original_native_Source_bit_exact=True,state_all_zero=True,script_sha256=sha(__file__),source_cache_sha256=sha(p),trace_sha256=sha(tp),seconds=time.perf_counter()-start,note='LayerNorm statistics are input/channel population variances from detached activations. Association with large Jacobian is not an ablation or causal proof.')
 out=dict(provenance=provenance,records=records,native_node55=np.asarray(c['Source'])[55].tolist(),corrected_node55=np.asarray(c['E'])[55].tolist())
 torch.save(dict(raw=raw,actual_initial_boxes=actual,GT_read=False,provenance=provenance),O/'layer_probe_raw.pt');write(O/'layer_probe.json',out);write(O/'layer_probe_sealed.json',dict(status='sealed_before_interpretation',GT_read=False,script_sha256=sha(__file__),files={k:sha(O/k)for k in ['layer_probe_raw.pt','layer_probe.json']}));print(json.dumps(provenance));print(json.dumps([r for r in records if r['view']==1 and r['kind']=='LayerNorm'],indent=2))
finally:
 for h in handles:h.remove()
 del rp,model;gc.collect();torch.cuda.empty_cache();lock.close()
