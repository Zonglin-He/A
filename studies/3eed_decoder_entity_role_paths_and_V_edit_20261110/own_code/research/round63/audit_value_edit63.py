"""Independent CPU-SVD direction construction and functional decoder replay."""
import argparse,sys,time,torch
import torch.nn.functional as F
from common63 import *
sys.path.insert(0,str(ROOT/'research/round62'));from audit62 import readout
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def mha(m,q,k,v,mask=None,W=None,b=None):
 return F.multi_head_attention_forward(q,k,v,288,8,m.in_proj_weight if W is None else W,m.in_proj_bias if b is None else b,None,None,False,.1,m.out_proj.weight,m.out_proj.bias,training=False,key_padding_mask=mask,need_weights=True)
def permuted(sid,mask,padding):
 good=np.flatnonzero(~padding.reshape(-1));shift=1+int(hashlib.sha256(('R63_value_direction_v2|'+sid).encode()).hexdigest()[:12],16)%(len(good)-1);ix=np.arange(len(padding.reshape(-1)));ix[good]=np.roll(good,shift);return mask[ix]
def matrices(model,z,v,cfg,sm,rm,valid):
 m=model.decoder[cfg['selected_layer']].cross_l;lo=576+36*cfg['selected_head'];hi=lo+36;W=m.in_proj_weight.detach();b=m.in_proj_bias.detach();M=np.column_stack([W[lo:hi].cpu().numpy(),b[lo:hi].cpu().numpy()]).astype(float);X=np.column_stack([z[v+'_prefix_lang'][0].astype(float),np.ones(len(sm))]);erase=rm if cfg['selected_kind']=='drop_reference' else sm;protect=sm if cfg['selected_kind']=='drop_reference' else rm;d=np.zeros_like(M);random=d.copy();err=0.
 if valid:
  _,S,Vh=np.linalg.svd(X[protect],full_matrices=False);B=Vh[S>S[0]*1e-6].T
  def rb(mask):
   rows=X[mask];res=rows-(rows@B)@B.T;_,ss,vh=np.linalg.svd(res,full_matrices=False);return vh[ss>ss[0]*1e-6].T if ss[0]>=1e-6 else np.empty((289,0))
  U=rb(erase)
  if U.shape[1]:
   d=-(M@U)@U.T;RU=rb(permuted(z['_sid'],erase,z[v+'_prefix_text_padding']))
   if RU.shape[1]:random=-(M@RU)@RU.T;random*=np.linalg.norm(d)/max(np.linalg.norm(random),1e-12)
 patches={}
 for mode,D in [('physical',d),('opposite',-d),('shuffled',random)]:
  new=(M+D).astype(np.float32);nw=W.clone();nb=b.clone();nw[lo:hi]=torch.tensor(new[:,:288],device='cuda');nb[lo:hi]=torch.tensor(new[:,288],device='cuda');patches[mode]=(nw,nb)
  if valid:
   old=X[protect]@M.T;changed=X[protect]@new.astype(float).T;err=max(err,float(np.max(np.abs(old-changed))))
 assert err<1e-4;return patches,err
def random_path(m,q,lang,padding,sm,rm,head,erasekind,sid):
 Q,K,V=F._in_projection_packed(q,lang,lang,m.in_proj_weight,m.in_proj_bias);Q=Q.reshape(-1,1,8,36).permute(1,2,0,3)[0];K=K.reshape(-1,1,8,36).permute(1,2,0,3)[0];V=V.reshape(-1,1,8,36).permute(1,2,0,3)[0];a=((Q/6)@K.transpose(-1,-2)).masked_fill(padding[0][None,None],-torch.inf).softmax(-1);erase=rm if erasekind=='drop_reference' else sm;other=permuted(sid,erase,padding.cpu().numpy());em=torch.tensor(erase,device='cuda');om=torch.tensor(other,device='cuda');target=a[head,:,em]@V[head,em];random=a[head,:,om]@V[head,om];W=m.out_proj.weight[:,head*36:(head+1)*36];ct=F.linear(target,W,None);rc=F.linear(random,W,None);change=rc*ct.norm(dim=-1,keepdim=True)/rc.norm(dim=-1,keepdim=True).clamp_min(1e-12);return change[:,None]
def forward(model,z,v,cfg,sm,rm,valid,mode,patches,source=None):
 t=lambda k:torch.tensor(z[v+'_'+k],device='cuda');h=t('prefix_query');pos=t('prefix_pos');vision=t('prefix_vis').transpose(0,1);lang=t('prefix_lang').transpose(0,1);padding=t('prefix_text_padding');anchor=t('query_anchor_xyz')[None];saved={}
 for i,d in enumerate(model.decoder):
  pp=d.self_posembed(pos).transpose(1,2).contiguous().transpose(0,1);q=h.transpose(0,1);q=d.norm1(q+mha(d.self_attn,q+pp,q+pp,q)[0]);W=b=None
  if i==cfg['selected_layer'] and mode not in ['source','random_path'] and valid:W,b=patches[mode if mode in ['opposite','shuffled'] else 'physical']
  o=mha(d.cross_l,q+pp,lang,lang,padding,W,b)[0]
  if i==cfg['selected_layer'] and mode=='random_path' and valid:o=o-random_path(d.cross_l,q+pp,lang,padding,sm,rm,cfg['selected_head'],cfg['selected_kind'],z['_sid'])
  q=d.norm_l(q+o);visual=mha(d.cross_v,q+pp,vision,vision)[0]
  if i==cfg['selected_layer']:
   if mode=='source':saved={'text':q.detach().clone(),'visual':visual.detach().clone()}
   if mode=='text_only':visual=source['visual']
   if mode=='visual_only':q=source['text']
  q=d.norm_v(q+visual);q=d.norm2(q+d.ffn(q));h=q.transpose(0,1).contiguous();ph=model.prediction_heads[i];hh=h.transpose(1,2).contiguous();pos=torch.cat([anchor+ph.center_residual_head(hh).transpose(2,1),ph.size_pred_head(hh).transpose(2,1).reshape(1,256,3)],-1)
 return h[0].cpu().numpy(),pos[0].cpu().numpy(),saved
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=model.contrastive_align_projection_image.cpu();cfg=read(ROOT/p['runtime_config']);root=ROOT/p['prediction_root'];ps=read(root/'SEALED.json');cap=ROOT/p['capture_root'];cs=read(cap/'SEALED.json');rolemap={}
 for r in read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json'):rolemap.setdefault(r['id_PRIVATE'],{})[r['role']]=np.asarray(r['mask'],bool)
 start=time.perf_counter();mxh=mxs=mxb=mxp=0.;checks=0
 for idx,row in enumerate(read(ROOT/p['manifest'])):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(root/rel)==ps['files'][rel];state_rel='states_PRIVATE/'+sid+'.npz';assert sha(root/state_rel)==ps['files'][state_rel];z=load(cap/rel);z['_sid']=sid;pred=load(root/rel);st=load(root/state_rel);g=load(ROOT/p['association_root']/rel);sm=rolemap[sid]['subject'];rm=rolemap[sid]['reference'];valid=bool(sm.any() and rm.any() and not (sm&rm).any());patches={};sources={};hs={}
  for v in 'ABC':
   patches[v],e=matrices(model,z,v,cfg,sm,rm,valid);mxp=max(mxp,e);hs[v],_,sources[v]=forward(model,z,v,cfg,sm,rm,valid,'source',patches[v]);assert abs(hs[v]-z[v+'_h']).max()<1e-4
  abc,r59=readout(head,hs,z,g);expected={'frozen_ABC':abc,'r59_direct_all':r59}
  for mode in ['physical','functional','opposite','shuffled','text_only','visual_only','random_path']:
   hs={}
   for v in 'ABC':
    h,b,_=forward(model,z,v,cfg,sm,rm,valid,mode,patches[v],sources[v]);hs[v]=h;eh=float(abs(h-st[mode+'_'+v+'_h']).max());eb=float(abs(b-st[mode+'_'+v+'_boxes']).max());mxh=max(mxh,eh);mxb=max(mxb,eb);assert eh<1e-4 and eb<1e-4,(idx,mode,v,eh,eb)
   native,fused=readout(head,hs,z,g);expected[mode]=native
   if mode=='physical':expected['physical_R59']=fused
  old=load(ROOT/p['heads_root']/rel);assert np.array_equal(pred['scores'][p['arm_names'].index('head_path_knockout')],old['scores'][list(old['arm_names']).index('H'+str(cfg['selected_head']))])
  for arm,scores in expected.items():j=p['arm_names'].index(arm);e=float(abs(scores-pred['scores'][j]).max());mxs=max(mxs,e);assert e<3e-5 and scores.argmax()==pred['queries'][j],(idx,arm,e);checks+=1
  assert np.array_equal(pred['source_boxes'],z['A_boxes']) and np.array_equal(pred['selected_original_boxes'],z['A_boxes'][pred['queries']])
  if (idx+1)%16==0:print('AUDIT_R63_VALUE',idx+1,336,flush=True)
 assert fingerprint(model)==fp;out=ROOT/p['audit_root'];assert not out.exists();out.mkdir();result={'state':'PASS_R63_VALUE_DIRECTION_INDEPENDENT_CPU_SVD_FUNCTIONAL_REPLAY','n':336,'independent_readouts':checks,'max_score_error':mxs,'max_hidden_error':mxh,'max_internal_box_error':mxb,'max_protected_value_change':mxp,'all_Top1_exact':True,'prediction_seal_sha256':sha(root/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
