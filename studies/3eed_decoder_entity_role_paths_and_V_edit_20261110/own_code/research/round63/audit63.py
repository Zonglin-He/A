"""Independent functional replay: modify projected V or role A, not output paths."""
import argparse,sys,time,torch
import torch.nn.functional as F
from common63 import *
sys.path.insert(0,str(ROOT/'research/round62'));from audit62 import readout
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def mha(m,q,k,v,mask=None):
 return F.multi_head_attention_forward(q,k,v,288,8,m.in_proj_weight,m.in_proj_bias,None,None,False,.1,m.out_proj.weight,m.out_proj.bias,training=False,key_padding_mask=mask,need_weights=True)[0]
def role_mha(m,q,lang,padding,sm,rm,kind,head=None,diagnostic=False):
 qq,kk,vv=F._in_projection_packed(q,lang,lang,m.in_proj_weight,m.in_proj_bias)
 qh=qq.transpose(0,1).reshape(1,len(q),8,36).permute(0,2,1,3)[0]
 kh=kk.transpose(0,1).reshape(1,len(lang),8,36).permute(0,2,1,3)[0]
 vh=vv.transpose(0,1).reshape(1,len(lang),8,36).permute(0,2,1,3)[0]
 a=(qh@kh.transpose(-1,-2)/6.).masked_fill(padding[0][None,None],-torch.inf).softmax(-1)
 sm=torch.as_tensor(sm,device=q.device);rm=torch.as_tensor(rm,device=q.device)
 cs=torch.matmul(a[:,:,sm],vh[:,sm]);cr=torch.matmul(a[:,:,rm],vh[:,rm]);ms=a[:,:,sm].sum(-1);mr=a[:,:,rm].sum(-1)
 trace=torch.stack([ms,mr,cs.norm(dim=-1),cr.norm(dim=-1),F.cosine_similarity(cs,cr,dim=-1)]).cpu().numpy()
 selected=list(range(8)) if head is None else [head]
 if kind in ['drop_subject','drop_reference']:
  vnew=vh.clone();mask=sm if kind=='drop_subject' else rm
  for h in selected:vnew[h,mask]=0
  vh=vnew
 elif kind=='swap_roles':
  an=a.clone()
  for h in selected:
   good=(ms[h]>1e-8)&(mr[h]>1e-8);sc=torch.where(good,mr[h]/ms[h].clamp_min(1e-8),torch.ones_like(ms[h]));rc=torch.where(good,ms[h]/mr[h].clamp_min(1e-8),torch.ones_like(ms[h]));an[h,:,sm]=a[h,:,sm]*sc[:,None];an[h,:,rm]=a[h,:,rm]*rc[:,None]
  assert abs(an.sum(-1)-a.sum(-1)).max()<1e-5;a=an
 elif kind is not None:raise ValueError(kind)
 out=(a@vh).permute(1,0,2).reshape(len(q),1,288);return F.linear(out,m.out_proj.weight,m.out_proj.bias),trace
def forward(model,z,v,sm,rm,valid,layer=None,kind=None,head=None,trace=False):
 t=lambda k:torch.tensor(z[v+'_'+k],device='cuda');h=t('prefix_query');pos=t('prefix_pos');vision=t('prefix_vis').transpose(0,1);language=t('prefix_lang').transpose(0,1);padding=t('prefix_text_padding');anchor=t('query_anchor_xyz')[None];traces={}
 for i,d in enumerate(model.decoder):
  pp=d.self_posembed(pos).transpose(1,2).contiguous().transpose(0,1);q=h.transpose(0,1);q=d.norm1(q+mha(d.self_attn,q+pp,q+pp,q));o=mha(d.cross_l,q+pp,language,language,padding)
  if trace or (i==layer and valid):
   alt,tr=role_mha(d.cross_l,q+pp,language,padding,sm,rm,kind if i==layer and valid else None,head)
   if trace:traces[i]=tr
   if i==layer and valid:o=alt
  q=d.norm_l(q+o);q=d.norm_v(q+mha(d.cross_v,q+pp,vision,vision));q=d.norm2(q+d.ffn(q));h=q.transpose(0,1).contiguous();ph=model.prediction_heads[i];hh=h.transpose(1,2).contiguous();center=anchor+ph.center_residual_head(hh).transpose(2,1);size=ph.size_pred_head(hh).transpose(2,1).reshape(1,256,3);pos=torch.cat([center,size],-1)
 return h[0].cpu().numpy(),pos[0].cpu().numpy(),traces
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=model.contrastive_align_projection_image.cpu();root=ROOT/p['prediction_root'];cap=ROOT/p['capture_root'];predseal=read(root/'SEALED.json');capseal=read(cap/'SEALED.json');roles=read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json');rolemap={}
 for r in roles:rolemap.setdefault(r['id_PRIVATE'],{})[r['role']]=np.asarray(r['mask'],bool)
 maxscore=maxhidden=maxbox=maxtrace=0.;checks=0;start=time.perf_counter()
 for i,row in enumerate(read(ROOT/p['manifest'])):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(root/rel)==predseal['files'][rel] and sha(cap/rel)==capseal['files'][rel];state_rel='states_PRIVATE/'+sid+'.npz';assert sha(root/state_rel)==predseal['files'][state_rel];z=load(cap/rel);pred=load(root/rel);st=load(root/state_rel);sm=rolemap[sid]['subject'];rm=rolemap[sid]['reference'];valid=bool(sm.any() and rm.any() and not (sm&rm).any());G=ROOT/p['association_root'];assert sha(G/rel)==read(G/'SEALED.json')['files'][rel];g=load(G/rel)
  if p['mode']=='screen':
   hs={};trace_rel='traces_PRIVATE/'+sid+'.npz';assert sha(root/trace_rel)==predseal['files'][trace_rel];savedtr=load(root/trace_rel)
   for v in 'ABC':
    hs[v],_,tr=forward(model,z,v,sm,rm,valid,trace=True)
    for l,x in tr.items():e=float(abs(x-savedtr[v+'_L'+str(l)]).max());maxtrace=max(maxtrace,e);assert e<1e-4
   abc,r59=readout(head,hs,z,g)
   for name,scores in [('frozen_ABC',abc),('r59_direct_all',r59)]:
    j=p['arm_names'].index(name);e=float(abs(scores-pred['scores'][j]).max());maxscore=max(maxscore,e);assert e<3e-5 and scores.argmax()==pred['queries'][j];checks+=1
  for arm in p['arm_names'][3:]:
   if p['mode']=='screen':layer=int(arm[1]);kind=arm[3:];hh=None
   else:layer=p['selected_layer'];kind=p['selected_kind'];hh=int(arm[1:])
   hs={}
   for v in 'ABC':
    h,b,_=forward(model,z,v,sm,rm,valid,layer,kind,hh);hs[v]=h;eh=float(abs(h-st[arm+'_'+v+'_h']).max());eb=float(abs(b-st[arm+'_'+v+'_boxes']).max());maxhidden=max(maxhidden,eh);maxbox=max(maxbox,eb);assert eh<1e-4 and eb<1e-4,(i,arm,v,eh,eb)
   s,_=readout(head,hs,z,g);j=p['arm_names'].index(arm);e=float(abs(s-pred['scores'][j]).max());maxscore=max(maxscore,e);assert e<3e-5 and s.argmax()==pred['queries'][j],(i,arm,e);checks+=1
  assert np.array_equal(pred['source_boxes'],z['A_boxes']) and np.array_equal(pred['queries'],pred['scores'].argmax(1)) and np.array_equal(pred['selected_original_boxes'],z['A_boxes'][pred['queries']])
  if (i+1)%16==0:print('AUDIT_R63_'+p['mode'],i+1,336,flush=True)
 assert fingerprint(model)==fp;out=ROOT/p['audit_root'];assert not out.exists();out.mkdir();result={'state':'PASS_R63_'+p['mode'].upper()+'_INDEPENDENT_ROLE_VALUE_OR_ATTENTION_EDIT_REPLAY','n':336,'independent_readouts':checks,'max_score_error':maxscore,'max_hidden_error':maxhidden,'max_internal_box_error':maxbox,'max_native_head_role_trace_error':maxtrace,'all_Top1_exact':True,'prediction_seal_sha256':sha(root/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
