"""Conditional single-site semantic lesions with controlled branch norm.

Not a learned patch: no labels, pseudo-target fitting or adaptive strength.
Native Q/K stay fixed for value-only lesions. Joint KV permutation is only
an invariance control. Increment controls retain the original source branch.
"""
import argparse,sys,time,torch
from common62 import *
import compute62 as m
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def permutation(sid,L,mask):
 valid=np.flatnonzero(~np.asarray(mask,bool).reshape(-1));assert len(valid)>1;shift=1+int(hashlib.sha256(('R62_semantic_value_v1|'+sid).encode()).hexdigest()[:12],16)%(len(valid)-1);out=np.arange(L);out[valid]=np.roll(valid,shift);assert (out[valid]!=valid).all();return out
def normmatch(x,base):
 target=base.norm(dim=-1,keepdim=True);n=x.norm(dim=-1,keepdim=True);bad=n<=1e-12;out=x*target/n.clamp_min(1e-12);out=torch.where(bad,base,out);return out,int(bad.sum())
def semantic_chain(model,z,v,site,kind,sid):
 module=model.decoder[site].cross_l;cache={};info={};attention={};handles=[]
 def pre(_m,args,kwargs):cache.update(kwargs)
 def capture_last(_m,args,out):attention['last']=out[1][0].detach().cpu().numpy()
 def after(_m,args,out):
  base=out[0];q,k,value=cache['query'],cache['key'],cache['value'];mask=cache.get('key_padding_mask');perm=permutation(sid,len(k),mask.detach().cpu().numpy());ix=torch.tensor(perm,device=q.device);kwargs=dict(attn_mask=cache.get('attn_mask'),key_padding_mask=mask,need_weights=True)
  if kind=='increment_uniform':
   d=module.embed_dim;projected=torch.nn.functional.linear(value,module.in_proj_weight[2*d:],module.in_proj_bias[2*d:]);valid=(~mask[0]).nonzero().flatten();mean=projected[valid].mean(0,keepdim=True).expand(len(q),-1,-1);alt=torch.nn.functional.linear(mean,module.out_proj.weight,module.out_proj.bias)
  elif kind=='increment_joint':
   kwargs['key_padding_mask']=mask[:,ix];alt=torch.nn.MultiheadAttention.forward(module,q,k[ix],value[ix],**kwargs)[0]
  else:alt=torch.nn.MultiheadAttention.forward(module,q,k,value[ix],**kwargs)[0]
  matched,unknown=normmatch(alt,base);cos=torch.nn.functional.cosine_similarity(base,matched,dim=-1);info['median_source_vs_alternative_cosine']=float(cos.median());info['median_relative_alternative_vector_change']=float(((matched-base).norm(dim=-1)/base.norm(dim=-1).clamp_min(1e-12)).median());info['zero_alternative_norm_count']=unknown
  if kind=='value1':new=matched;target=base.norm(dim=-1)
  elif kind=='value15':new=1.5*matched;target=1.5*base.norm(dim=-1)
  else:
   new,zeros=normmatch(base+.5*matched,1.5*base);info['zero_total_increment_norm_count']=zeros;target=1.5*base.norm(dim=-1)
  info['max_relative_branch_norm_error']=float(((new.norm(dim=-1)-target).abs()/target.clamp_min(1e-12)).max());attention['selected']=out[1][0].detach().cpu().numpy();return new,out[1]
 handles=[module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(after),model.decoder[5].cross_l.register_forward_hook(capture_last)]
 try:h,hist,_=m.decoder_chain(model,z,v)
 finally:
  for handle in handles:handle.remove()
 return h,hist,info,attention
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);selection=read(ROOT/p['selection']);site=selection['selected_layer'];assert site is not None;m.gpu_strict();rows=read(ROOT/p['manifest']);cap=ROOT/p['capture_root'];cs=read(cap/'SEALED.json');model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=m.cpu_head(load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz'));out=ROOT/p['semantics_root'];assert not out.exists();out.mkdir(parents=True)
 for folder in ['samples_PRIVATE','states_PRIVATE','cases_PRIVATE','attention_PRIVATE']:(out/folder).mkdir()
 start=time.perf_counter();statuses=[];maxnorm=0.;maxjoint=0.;jointqueries=0
 try:
  for i,row in enumerate(rows):
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel];z=load(cap/rel);G=ROOT/p['association_root'];assert sha(G/rel)==read(G/'SEALED.json')['files'][rel];g=load(G/rel);mainroot=ROOT/p['layers_root'];old=load(mainroot/rel);assert sha(mainroot/rel)==read(mainroot/'SEALED.json')['files'][rel];result={'frozen_ABC':old['scores'][list(old['arm_names']).index('frozen_ABC')],'selected_native15':old['scores'][list(old['arm_names']).index('native_l'+str(site))]};details={};states={};att={}
   # Reconstruct both source and correct edit for offline role diagnostics.
   for name,layer in [('source',None),('native',site)]:
    hs={}
    for v in 'ABC':
     h,hist,aa=m.decoder_chain(model,z,v,layer,return_attention=True);hs[v]=h;states[name+'_'+v+'_h']=h;att[name+'_'+v+'_selected']=aa[site];att[name+'_'+v+'_last']=aa[5]
    native,_,_=m.readouts(head,hs,z,g)
    expected=result['frozen_ABC' if name=='source' else 'selected_native15'];assert abs(native-expected).max()<3e-5 and native.argmax()==expected.argmax()
   for kind in ['value1','value15','increment_value','increment_uniform','increment_joint']:
    hs={};bb={};record={}
    for v in 'ABC':
     h,hist,info,aa=semantic_chain(model,z,v,site,kind,sid);hs[v]=h;bb[v]=hist[-1];states[kind+'_'+v+'_h']=h;record[v]=info;maxnorm=max(maxnorm,info['max_relative_branch_norm_error'])
    arm='selected_'+kind;result[arm]=m.readouts(head,hs,z,g)[0];record['identity']=m.lineage(z,bb,g);details[arm]=record
   jointdiff=float(abs(result['selected_increment_joint']-result['selected_native15']).max());maxjoint=max(maxjoint,jointdiff);jointqueries+=int(result['selected_increment_joint'].argmax()!=result['selected_native15'].argmax());scores=np.stack([result[arm] for arm in SEMANTIC_ARMS]);q=scores.argmax(1);assert np.isfinite(scores).all();np.savez_compressed(out/rel,arm_names=np.array(SEMANTIC_ARMS),scores=scores,queries=q,source_boxes=z['A_boxes'],source_scores=z['A_scores'],selected_original_boxes=z['A_boxes'][q]);np.savez_compressed(out/'states_PRIVATE'/(sid+'.npz'),**states);np.savez_compressed(out/'attention_PRIVATE'/(sid+'.npz'),**att);dump(out/'cases_PRIVATE'/(sid+'.json'),details);statuses.append({'id':sid,'arm_statuses':{arm:{'unknown':False,'fallback':False,'no_parameter_update':True} for arm in SEMANTIC_ARMS}})
   if (i+1)%16==0:print('R62_SEMANTICS',i+1,336,flush=True)
  assert fingerprint(model)==fp;cost={'n':336,'selected_layer':site,'arms':len(SEMANTIC_ARMS),'source_restored_exact':True,'no_optimizer_or_gate_or_label_fit':True,'max_relative_branch_norm_error':maxnorm,'joint_positive_max_score_error':maxjoint,'joint_positive_query_changes':jointqueries,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);dump(out/'READOUT_STATUS_PRIVATE.json',statuses);seal(out,'R62_SELECTED_SITE_SEMANTIC_CONTROLS_SEALED_BEFORE_SCORE',n=336,arm_names=SEMANTIC_ARMS,selection_sha256=sha(ROOT/p['selection']));print(cost,flush=True)
 except BaseException as error:dump(out/'FAILED.json',{'type':type(error).__name__,'error':str(error),'completed':len(statuses),'automatic_retry':False});raise
if __name__=='__main__':main()
