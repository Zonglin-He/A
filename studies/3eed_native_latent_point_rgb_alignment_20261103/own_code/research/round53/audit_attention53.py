"""Independent Q/K equation check on four hash-selected component replays."""
import argparse,hashlib,json,sys,time,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,fingerprint,PREFIX_KEYS
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 import torch
 import torch.nn.functional as F
 torch.manual_seed(0);np.random.seed(0);start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);rows=read(ROOT/p['manifest']);selected=[]
 for platform in ['drone','quad']:selected+=sorted([r for r in rows if r['platform']==platform],key=lambda r:hashlib.sha256(('R53_equation_audit_v1:'+r['id']).encode()).digest())[:2]
 folder=ROOT/p['capture_root'];seal=read(folder/'SEALED.json');errors=[];calls=0
 for row in selected:
  rel='samples_PRIVATE/'+row['id']+'.npz';assert sha(folder/rel)==seal['files'][rel]
  with np.load(folder/rel,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
  actual={};equations=[];handles=[]
  def before(key):
   def hook(module,args,kwargs):
    q=kwargs.get('query',args[0] if args else None);k=kwargs.get('key',args[1] if len(args)>1 else None)
    qw,kw,_=module.in_proj_weight.chunk(3);qb,kb,_=module.in_proj_bias.chunk(3);Q=F.linear(q,qw,qb).transpose(0,1);K=F.linear(k,kw,kb).transpose(0,1);B,N,C=Q.shape;heads=module.num_heads;d=C//heads
    Q=Q.view(B,N,heads,d).transpose(1,2);K=K.view(B,K.shape[1],heads,d).transpose(1,2);scores=(Q/math.sqrt(d))@K.transpose(-1,-2)
    mask=kwargs.get('key_padding_mask')
    if mask is not None:scores=scores.masked_fill(mask[:,None,None],-torch.inf)
    actual[key+'_formula']=scores.softmax(-1).mean(1)
   return hook
  def after(key):
   def hook(module,args,value):actual[key]=value[1].detach()
   return hook
  modules={**{'token_point_'+str(i):x.cross_layer.cross_lv for i,x in enumerate(model.cross_encoder.layers)},'query_point_5':model.decoder[5].cross_v,'query_token_5':model.decoder[5].cross_l}
  for key,m in modules.items():handles += [m.register_forward_pre_hook(before(key),with_kwargs=True),m.register_forward_hook(after(key))]
  tensor=lambda key:torch.as_tensor(arrays[key],device='cuda')
  try:
   with torch.no_grad():
    text=model.tokenizer.batch_encode_plus([row['caption']+' . not mentioned'],padding='longest',return_tensors='pt').to('cuda');tf=model.text_projector(model.text_encoder(**text).last_hidden_state);xyz=tensor('fp2_xyz')[None]
    vis,lang=model.cross_encoder(vis_feats=tensor('fp2_features').T[None].contiguous(),pos_feats=model.pos_embed(xyz).transpose(1,2).contiguous(),padding_mask=torch.zeros((1,1024),device='cuda',dtype=torch.bool),text_feats=tf,text_padding_mask=text.attention_mask.ne(1).bool(),end_points={},detected_feats=None,detected_mask=None);calls+=1
    assert torch.equal(vis,tensor('seed_features').T[None]) and torch.equal(lang,tensor('text_memory')[None])
    args=[tensor(k) if k in arrays else None for k in PREFIX_KEYS];result=model.decoder[5](*args,detected_feats=None,detected_mask=None);calls+=1;assert torch.equal(result[0],tensor('h'))
  finally:
   for h in handles:h.remove()
  for key in modules:
   equation=float((actual[key]-actual[key+'_formula']).abs().max());saved=float((actual[key][0]-tensor(key)).abs().max());assert equation<1e-6 and saved==0.,(key,equation,saved);errors.append({'module':key,'formula_max_abs_error':equation,'native_replay_max_abs_error':saved})
 assert fingerprint(model)==fp
 out=ROOT/'research/round53/attention_equation_audit_attempt001';assert not out.exists();out.mkdir()
 result={'state':'PASS_R53_INDEPENDENT_QK_EQUATION_AND_COMPONENT_REPLAY','n_hash_samples':len(selected),'rows_per_platform':2,'component_replays':calls,'new_full_grounder_forwards':0,'new_text_encoder_calls':4,'native_attention_matrices_checked':len(errors),'max_formula_absolute_error':max(x['formula_max_abs_error'] for x in errors),'saved_native_replay_bitexact':True,'cross_encoder_features_replay_bitexact':True,'last_decoder_features_replay_bitexact':True,'model_unchanged':True,'optimizer_steps':0,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'capture_seal_sha256':sha(folder/'SEALED.json'),'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'errors':errors};(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='errors'}),flush=True)
if __name__=='__main__':main()
