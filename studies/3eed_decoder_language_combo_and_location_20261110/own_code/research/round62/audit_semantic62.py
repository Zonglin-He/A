"""Independent selected-site semantic lesion replay, not producer import."""
import argparse,sys,time,torch
import torch.nn.functional as F
from common62 import *
from audit62 import readout
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
def mha(m,q,k,v,mask):
 return F.multi_head_attention_forward(q,k,v,288,8,m.in_proj_weight,m.in_proj_bias,None,None,False,.1,m.out_proj.weight,m.out_proj.bias,training=False,key_padding_mask=mask,need_weights=True)[0]
def matchnorm(x,y):return torch.where(x.norm(dim=-1,keepdim=True)>1e-12,x*y.norm(dim=-1,keepdim=True)/x.norm(dim=-1,keepdim=True).clamp_min(1e-12),y)
def forward(model,z,v,site,kind,sid):
 t=lambda k:torch.tensor(z[v+'_'+k],device='cuda');q=t('prefix_query');pos=t('prefix_pos');vision=t('prefix_vis').transpose(0,1);language=t('prefix_lang').transpose(0,1);mask=t('prefix_text_padding');anchor=t('query_anchor_xyz')[None];normerr=0.;branch_diagnostic={}
 for i,d in enumerate(model.decoder):
  p=d.self_posembed(pos).transpose(1,2).contiguous().transpose(0,1);q=q.transpose(0,1);q=d.norm1(q+mha(d.self_attn,q+p,q+p,q,None));o=mha(d.cross_l,q+p,language,language,mask)
  if i==site and kind!='source':
   if kind=='native':o=o*1.5
   else:
    valid=np.flatnonzero(~mask.cpu().numpy().reshape(-1));shift=1+int(hashlib.sha256(('R62_semantic_value_v1|'+sid).encode()).hexdigest()[:12],16)%(len(valid)-1);perm=np.arange(len(language));perm[valid]=np.roll(valid,shift);ix=torch.tensor(perm,device='cuda');m=d.cross_l
    if kind=='increment_uniform':
     values=F.linear(language,m.in_proj_weight[576:],m.in_proj_bias[576:]);avg=values[torch.tensor(valid,device='cuda')].mean(0,keepdim=True);alt=F.linear(avg.expand(len(q),-1,-1),m.out_proj.weight,m.out_proj.bias)
    elif kind=='increment_joint':alt=mha(m,q+p,language[ix],language[ix],mask[:,ix])
    else:alt=mha(m,q+p,language,language[ix],mask)
    aligned=matchnorm(alt,o)
    if kind=='value1':new=aligned;target=o.norm(dim=-1)
    elif kind=='value15':new=1.5*aligned;target=1.5*o.norm(dim=-1)
    else:new=matchnorm(o+.5*aligned,1.5*o);target=1.5*o.norm(dim=-1)
    normerr=max(normerr,float(((new.norm(dim=-1)-target).abs()/target.clamp_min(1e-12)).max()));cos=F.cosine_similarity(new,o,dim=-1);scale=1. if kind=='value1' else 1.5;relative=(new-scale*o).norm(dim=-1)/(scale*o.norm(dim=-1)).clamp_min(1e-12);branch_diagnostic={'median_final_branch_cosine_to_native':float(cos.median()),'mean_final_branch_relative_vector_change':float(relative.mean()),'median_final_branch_relative_vector_change':float(relative.median()),'fraction_final_branch_relative_change_over_0p1':float((relative>.1).float().mean()),'max_final_branch_relative_vector_change':float(relative.max()),'zero_source_branch_norm_n':int((o.norm(dim=-1)<=1e-12).sum())};o=new
  q=d.norm_l(q+o);q=d.norm_v(q+mha(d.cross_v,q+p,vision,vision,None));q=d.norm2(q+d.ffn(q));q=q.transpose(0,1).contiguous();pred=model.prediction_heads[i];hh=q.transpose(1,2).contiguous();center=anchor+pred.center_residual_head(hh).transpose(2,1);size=pred.size_pred_head(hh).transpose(2,1).view(1,256,3);pos=torch.cat([center,size],-1)
 return q[0].cpu().numpy(),normerr,branch_diagnostic
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.set_grad_enabled(False);model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);head=model.contrastive_align_projection_image.cpu();rows=read(ROOT/p['manifest']);selection=read(ROOT/p['selection']);site=selection['selected_layer'];cap=ROOT/p['capture_root'];root=ROOT/p['prediction_root'];ps=read(root/'SEALED.json');cs=read(cap/'SEALED.json');start=time.perf_counter();maxscore=0.;maxh=0.;normerr=0.;checks=0;strength=[]
 for i,row in enumerate(rows):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(root/rel)==ps['files'][rel];z=load(cap/rel);pred=load(root/rel);st=load(root/'states_PRIVATE'/(sid+'.npz'));G=ROOT/p['association_root'];assert sha(G/rel)==read(G/'SEALED.json')['files'][rel];g=load(G/rel);old=load(ROOT/p['layers_root']/rel)
  for arm,key in [('frozen_ABC','frozen_ABC'),('selected_native15','native_l'+str(site))]:assert np.array_equal(pred['scores'][SEMANTIC_ARMS.index(arm)],old['scores'][list(old['arm_names']).index(key)])
  for kind,name in [('source','frozen_ABC'),('native','selected_native15')]+[(k,'selected_'+k) for k in ['value1','value15','increment_value','increment_uniform','increment_joint']]:
   hs={}
   for v in 'ABC':
    h,e,bd=forward(model,z,v,site,kind,sid);hs[v]=h
    if bd:strength.append({'row':f'DEV336_{i+1:05d}','platform':row['platform'],'view':v,'arm':name,**bd})
    normerr=max(normerr,e);stored=st[kind+'_'+v+'_h'];error=float(abs(h-stored).max());maxh=max(maxh,error);assert error<1e-4
   scores,_=readout(head,hs,z,g);j=SEMANTIC_ARMS.index(name);e=float(abs(scores-pred['scores'][j]).max());maxscore=max(maxscore,e);assert e<3e-5 and scores.argmax()==pred['queries'][j],(i,name,e);checks+=1
  assert np.array_equal(pred['selected_original_boxes'],pred['source_boxes'][pred['queries']]) and np.array_equal(pred['source_boxes'],z['A_boxes'])
  if (i+1)%32==0:print('AUDIT_R62_SEMANTICS',i+1,336,flush=True)
 assert fingerprint(model)==fp;out=ROOT/p['audit_root'];assert not out.exists();out.mkdir(parents=True);result={'state':'PASS_R62_SEMANTICS_INDEPENDENT_VALUE_ALIGNMENT_AND_NORM_CONTROL_AUDIT','n':336,'independent_readouts':checks,'max_score_error':maxscore,'max_hidden_error':maxh,'max_relative_branch_norm_error':normerr,'all_Top1_exact':True,'prediction_seal_sha256':sha(root/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);dump(out/'FINAL_BRANCH_STRENGTH_PUBLIC_SAFE.json',strength);print(result)
if __name__=='__main__':main()
