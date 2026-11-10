"""Independent layer-location replay with FP32 matrix-scaled functional MHA."""
import argparse,sys,time,torch
import torch.nn.functional as F
from common62 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def attn(m,q,k,v,mask=None,scale=1.):
 return F.multi_head_attention_forward(q,k,v,288,8,m.in_proj_weight,m.in_proj_bias,None,None,False,.1,m.out_proj.weight if scale==1 else m.out_proj.weight*scale,m.out_proj.bias if scale==1 else m.out_proj.bias*scale,training=False,key_padding_mask=mask,need_weights=True)[0]
def chain(model,z,v,edited=None):
 t=lambda k:torch.tensor(z[v+'_'+k],device='cuda');h=t('prefix_query');pos=t('prefix_pos');vision=t('prefix_vis').transpose(0,1);language=t('prefix_lang').transpose(0,1);mask=t('prefix_text_padding');anchor=t('query_anchor_xyz')[None]
 for i in range(6):
  d=model.decoder[i];pp=d.self_posembed(pos).transpose(1,2).contiguous().transpose(0,1);q=h.transpose(0,1);q=d.norm1(q+attn(d.self_attn,q+pp,q+pp,q));o=attn(d.cross_l,q+pp,language,language,mask,1.5 if i==edited else 1.);q=d.norm_l(q+o);q=d.norm_v(q+attn(d.cross_v,q+pp,vision,vision));q=d.norm2(q+d.ffn(q));h=q.transpose(0,1).contiguous();pred=model.prediction_heads[i];center=anchor+pred.center_residual_head(h.transpose(1,2).contiguous()).transpose(2,1);size=pred.size_pred_head(h.transpose(1,2).contiguous()).transpose(2,1).view(1,256,3);pos=torch.cat([center,size],-1)
 return h[0].cpu().numpy(),pos[0].cpu().numpy()
def score(raw,z,g):
 parts=[]
 for v in 'ABC':
  q=F.normalize(torch.tensor(np.asarray(raw[v],np.float32))[None],dim=-1);t=torch.tensor(z[v+'_tokens'])[None];mask=torch.tensor(z[v+'_token_mask']).bool();l=q@t.transpose(-1,-2)/.07;y=(torch.logsumexp(l[:,:,mask],-1)-torch.logsumexp(l[:,:,~mask],-1))[0];p=F.softmax(y.double(),0).numpy()
  if v!='A':
   known=g[v+'_W'].dot(p);mass=float(g[v+'_fixed_mass']);p=mass*known/max(float(known.sum()),1e-300)+(1-mass)*g['base']
  parts.append(p)
 return np.array(parts).sum(0)/3
def readout(head,hs,z,g):
 raw32={};raw64={}
 W=head[4].weight.numpy().astype(float);bias=head[4].bias.numpy().astype(float)
 for v in 'ABC':
  x=F.relu(F.linear(torch.tensor(hs[v])[None],head[0].weight,head[0].bias));x=F.relu(F.linear(x,head[2].weight,head[2].bias))[0];raw32[v]=F.linear(x[None],head[4].weight,head[4].bias)[0].numpy();raw64[v]=np.column_stack([x.numpy().astype(float),np.ones(256)])@np.column_stack([W,bias]).T
 def norm(x,fallback=None):
  d=np.linalg.norm(x,axis=-1,keepdims=True);a=np.divide(x,d,out=np.zeros_like(x),where=d>1e-12);return np.where(d>1e-12,a,fallback) if fallback is not None else a
 dirs={v:norm(raw64[v]) for v in 'ABC'};B=g['B_W'];mass=B.sum(1);conditional=np.zeros_like(dirs['A']);ok=mass>0;conditional[ok]=(B@dirs['B'])[ok]/mass[ok,None];r=g['B_affinity'].max(1);canonical=norm(dirs['A']+r[:,None]*conditional,dirs['A']);targets={'A':np.linalg.norm(raw64['A'],axis=1)[:,None]*canonical};c={'A':r}
 for v in 'BC':
  T=g[v+'_W'];d=T.sum(0);mapped=np.zeros_like(canonical);ok=d>0;mapped[ok]=(T.T@canonical)[ok]/d[ok,None];valid=np.linalg.norm(mapped,axis=1)>1e-12;c[v]=g[v+'_affinity'].max(0)*valid;targets[v]=np.linalg.norm(raw64[v],axis=1)[:,None]*norm(mapped,dirs[v])
 combined={v:(1-c[v][:,None])*raw32[v]+c[v][:,None]*targets[v] for v in 'ABC'};return score(raw32,z,g),score(combined,z,g)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_grad_enabled(False);torch.set_num_threads(4);model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=model.contrastive_align_projection_image.cpu();model.contrastive_align_projection_image=head # decoder parameters remain CUDA, matching head CPU only
 cap=ROOT/p['capture_root'];root=ROOT/p['prediction_root'];cs=read(cap/'SEALED.json');ps=read(root/'SEALED.json');rows=read(ROOT/p['manifest']);mode=p['mode'];arms=ps['arm_names'];start=time.perf_counter();maxscore=0.;maxh=0.;maxbox=0.;checks=0
 for i,row in enumerate(rows):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(root/rel)==ps['files'][rel];z=load(cap/rel);pred=load(root/rel);st=load(root/'states_PRIVATE'/(sid+'.npz'));G=ROOT/p['association_root'];assert sha(G/rel)==read(G/'SEALED.json')['files'][rel];g=load(G/rel);expected={}
  if mode=='combo':
   hs={v:chain(model,z,v)[0] for v in 'ABC'};expected['zero_ABC'],expected['zero_R59']=readout(head,hs,z,g);layers=[5]
   old=load(ROOT/p['r59_root']/rel);prior=load(ROOT/p['r61_root']/rel)
   for arm,k in [('source','source'),('frozen_ABC','frozen_ABC'),('r59_direct_all','direct_all')]:assert np.array_equal(pred['scores'][arms.index(arm)],old['scores'][list(old['arm_names']).index(k)])
   assert np.array_equal(pred['scores'][arms.index('r61_last15')],prior['scores'][list(prior['arm_names']).index('dec_language_boost')])
  else:
   old=load(ROOT/p['combo_root']/rel);layers=list(range(5))
   for arm in COMBO_ARMS:assert np.array_equal(pred['scores'][arms.index(arm)],old['scores'][list(old['arm_names']).index(arm)])
  for layer in layers:
   hs={}
   for v in 'ABC':
    h,b=chain(model,z,v,layer);hs[v]=h;eh=float(abs(h-st['layer'+str(layer)+'_'+v+'_h']).max());eb=float(abs(b-st['layer'+str(layer)+'_'+v+'_internal_boxes']).max());maxh=max(maxh,eh);maxbox=max(maxbox,eb);assert eh<1e-4 and eb<1e-4
   native,fused=readout(head,hs,z,g);expected['native_l'+str(layer)]=native
   if layer==5:expected['R59_l5']=fused
  for arm,s in expected.items():
   j=arms.index(arm);e=float(abs(s-pred['scores'][j]).max());maxscore=max(maxscore,e);assert e<3e-5 and s.argmax()==pred['queries'][j],(i,arm,e);checks+=1
  assert np.array_equal(pred['queries'],pred['scores'].argmax(1)) and np.array_equal(pred['selected_original_boxes'],pred['source_boxes'][pred['queries']]) and np.array_equal(pred['source_boxes'],z['A_boxes']);assert z['A_query_points_sample_inds'].shape==(256,)
  if (i+1)%32==0:print('AUDIT_R62_'+mode,i+1,336,flush=True)
 # CPU head move changes device, not tensor values/fingerprint definition.
 assert fingerprint(model)==fp;out=ROOT/p['audit_root'];assert not out.exists();out.mkdir(parents=True);result={'state':'PASS_R62_'+mode.upper()+'_INDEPENDENT_CHAIN_FRESH_R59_AND_FIXED_BOX_AUDIT','n':336,'independent_readouts':checks,'max_score_error':maxscore,'max_hidden_error':maxh,'max_internal_box_error':maxbox,'all_Top1_exact':True,'prediction_seal_sha256':sha(root/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result,flush=True)
if __name__=='__main__':main()
