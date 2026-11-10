"""Post-result source-only control audit, no new predictions/edit/scoring."""
import sys,time,torch
import torch.nn.functional as F
from common63 import *
from paths_heads63 import role_masks,paths,gpu_strict
from target_value63 import permuted,patch
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
H=ROOT/'research/round63';p=read(H/'experiment/EDIT_FORMAL_EXECUTION_PLAN_LOCK.json');pin(p);assert read(H/'scalar_edit_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');cfg=read(ROOT/p['runtime_config']);assert cfg['selected_layer']==0;gpu_strict();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);rows=[];start=time.perf_counter();cap=ROOT/p['capture_root'];ss=read(cap/'SEALED.json');source_trace=ROOT/p['screen_root']
for idx,r in enumerate(read(ROOT/p['manifest'])):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==ss['files'][rel];z=load(cap/rel);z['_sid']=sid;sm,rm,valid,_=role_masks(sid,len(z['A_tokens']))
 for view in 'ABC':
  rec={'row':f'DEV336_{idx+1:05d}','platform':r['platform'],'view':view,'valid_roles':valid};saved={};module=model.decoder[0].cross_l
  def pre(m,args,kwargs):saved.update(kwargs)
  def post(m,args,out):
   q,k,v=saved['query'],saved['key'],saved['value'];padding=saved['key_padding_mask'];ms,mr,cs,cr,project,rebuilt=paths(m,q,k,v,padding,sm,rm);assert abs(rebuilt-out[0]).max()<1e-4
   if not valid:return
   head=cfg['selected_head'];lo=576+36*head;hi=lo+36;random=permuted(sid,rm,padding);_,_,rs,_,_,_=paths(m,q,k,v,padding,random,np.zeros_like(random));target=torch.zeros_like(cr);other=target.clone();target[head]=cr[head];other[head]=rs[head];ct=project(target)[:,0];rc=project(other)[:,0];cos=F.cosine_similarity(ct,rc,dim=-1);rec['median_reference_random_path_cosine']=float(cos.median());rec['random_token_fraction_from_reference']=float((random&rm).sum()/max(int(random.sum()),1))
   Q=F.linear(q,m.in_proj_weight[:288],m.in_proj_bias[:288]).reshape(-1,8,36)[:,head];K=F.linear(k,m.in_proj_weight[288:576],m.in_proj_bias[288:576]).reshape(-1,8,36)[:,head];V=F.linear(v,m.in_proj_weight[576:],m.in_proj_bias[576:]).reshape(-1,8,36)[:,head];A=((Q/6)@K.T).masked_fill(padding[0][None],-torch.inf).softmax(-1);random_t=torch.tensor(random,device=q.device);reference_t=torch.tensor(rm,device=q.device);intersection=torch.tensor(random&rm,device=q.device);rec['median_random_attention_mass_fraction_from_reference']=float((A[:,intersection].sum(-1)/A[:,random_t].sum(-1).clamp_min(1e-12)).median());valid_t=~padding[0];sv=torch.linalg.svdvals(V[valid_t].double());rec['V_uncentered_leading_singular_energy_fraction']=float(sv[0].square()/sv.square().sum());Wout=m.out_proj.weight[:,head*36:(head+1)*36];bias=F.linear(m.in_proj_bias[lo:hi],Wout,None);biaspath=mr[head,:,None]*bias[None];rec['median_reference_bias_path_norm_ratio']=float((biaspath.norm(dim=-1)/ct.norm(dim=-1).clamp_min(1e-12)).median())
   change,_,_,X,_=patch(model,z,view,cfg);M=torch.cat([m.in_proj_weight[lo:hi].double(),m.in_proj_bias[lo:hi,None].double()],1);new=(M+change).float();dV=F.linear(v[:,0],new[:,:288],new[:,288])-V;dr=F.linear(A[:,reference_t]@dV[reference_t],Wout,None);do=F.linear(A[:,~reference_t]@dV[~reference_t],Wout,None);rec['median_reference_path_delta_over_original']=float((dr.norm(dim=-1)/ct.norm(dim=-1).clamp_min(1e-12)).median());rec['median_nonreference_delta_over_reference_delta']=float((do.norm(dim=-1)/dr.norm(dim=-1).clamp_min(1e-12)).median());rec['median_reference_path_retained_norm_ratio']=float(((ct+dr).norm(dim=-1)/ct.norm(dim=-1).clamp_min(1e-12)).median())
  handles=[module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(post)]
  try:model.decoder[0](torch.tensor(z[view+'_prefix_query'],device='cuda'),torch.tensor(z[view+'_prefix_vis'],device='cuda'),torch.tensor(z[view+'_prefix_lang'],device='cuda'),torch.tensor(z[view+'_prefix_pos'],device='cuda'),None,torch.tensor(z[view+'_prefix_text_padding'],device='cuda'),detected_feats=None,detected_mask=None)
  finally:
   for h in handles:h.remove()
  rows.append(rec)
 if (idx+1)%64==0:print('CONTROL_SOURCE_ONLY',idx+1,336,flush=True)
assert fingerprint(model)==fp;known=[r for r in rows if r['valid_roles']];keys=[k for k in known[0] if type(known[0][k]) is float];summary={'state':'R63_POSTRESULT_DESCRIPTIVE_SOURCE_CONTROL_DIAGNOSTIC','n':336,'sample_views':len(rows),'valid_views':len(known),'no_new_counterfactual_predictions_or_task_scores':True,'source_weights_unchanged':True,'statistics_are_median_over_sample_view_query_medians_unless_named_token_or_energy_fraction':True,'median_statistics':{k:float(np.median([r[k] for r in known])) for k in keys},'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};out=H/'control_direction_diagnostic_attempt001';assert not out.exists();out.mkdir();dump(out/'ROWS_PUBLIC_SAFE.json',rows);dump(out/'SUMMARY_PUBLIC_SAFE.json',summary);seal(out,summary['state']);print(summary)
