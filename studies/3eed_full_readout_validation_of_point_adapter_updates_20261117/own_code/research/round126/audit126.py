"""NumPy64 full readout independent of producer Torch64 implementation."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round126';B=R/'crossplatform/round126';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lse(a,axis):
 m=a.max(axis=axis,keepdims=True);return (m+np.log(np.exp(a-m).sum(axis=axis,keepdims=True))).squeeze(axis)
def lp(h,u,mask,late,res,S):
 x=h.astype(float)
 for k in ['0','2']:x=np.maximum(x@S[k+'.weight'].astype(float).T+S[k+'.bias'].astype(float),0)
 x=x@S['4.weight'].astype(float).T+S['4.bias'].astype(float)+late.astype(float);q=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);s=q@u.astype(float).T/.07;m=mask.astype(bool);logits=lse(s[:,m],1)-lse(s[:,~m],1)+res.astype(float);return logits-lse(logits,0)
def soft(s):return np.exp(s-lse(s,0))
def main():
 mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==2304;reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=aliases=choices=bridges=precision_changes=0;mx=lossmx=0.;counts={k:{} for k in ['corr','wrong','random','free','corr013']}
 for key,seal in mat['directions'].items():
  co,name=key.split('/');src,_,tgt=name.split('_');D=B/co/name;assert sha(D/'SEALED.json')==seal
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  assert sha(R/col[src]['state_path'])==col[src]['state_sha256'];S0=np.load(R/col[src]['state_path']);S={k[6:]:S0[k] for k in S0.files if k.startswith('match.')}
  for e in [e for e in reg if e['cohort']==co and e['source']==src and e['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(a['arm_names'],a['scores']));old=R/'crossplatform/round125/formal'/co/name;orig=np.load(old/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(orig['arm_names'],orig['scores']));rec=rd(D/'records_PRIVATE'/(sid+'.json'));n+=1;assert np.array_equal(a['queries'],a['scores'].argmax(1))
   if e['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(p,op['prepared_A']) for p in po.values());continue
   ff=np.load(old/'features_PRIVATE'/(sid+'.npz'));meta=rd(old/'records_PRIVATE'/(sid+'.json'));late=ff['fixed_late'];mask=ff['zero_F_mask'];f=R/('crossplatform/round113' if co=='DEV' else 'crossplatform/round109')/name/'features_PRIVATE'/(sid+'.npz');assert str(f.relative_to(R))==rec['source_A_provider']['path'];assert sha(f)==rec['source_A_provider']['sha256'];provider=np.load(f);rA=provider['frozen_full_A_r' if co=='DEV' else 'real_A_r'];raw={}
   for setting in ['zero','corr1','corr3','wrong3','random3','free3']:raw[setting]=[lp(ff[setting+'_fit_'+v+'_h'],ff[setting+'_fit_'+v+'_u'],mask,late,rA,S) for v in ['A','B']]
   T=(np.exp(raw['zero'][0])+np.exp(raw['zero'][1]))/2;logT=np.log(T);loss={k:float((T*(logT-(a+b)/2)).sum())+meta['auxiliary'][k]['fit_keep'] for k,(a,b) in raw.items()}
   for k,v in loss.items():err=abs(v-rec['full_AB_objectives'][k]);lossmx=max(lossmx,err);assert err<1e-10
   for kind,opts in {'corr':['zero','corr3'],'wrong':['zero','wrong3'],'random':['zero','random3'],'free':['zero','free3'],'corr013':['zero','corr1','corr3']}.items():
    best=min(opts,key=lambda k:loss[k]);chosen=best if loss['zero']-loss[best]>max(1e-8,1e-6*abs(loss['zero'])) else 'zero';assert chosen==rec['choices'][kind];counts[kind][chosen]=counts[kind].get(chosen,0)+1;choices+=1
    for v in ['F','C']:want=op[('mean_context' if v=='F' else 'zero_C') if chosen=='zero' else chosen+'_'+v];assert np.array_equal(want,po['fullval_'+kind+'_'+v])
   rF=ff['fixed_r'];wanted={'full_teacher_bridge':soft(logT+rF.astype(float)-rA.astype(float))};pp=[]
   for v in ['A','B','C']:
    h=ff['teacher_'+v+'_h'] if v in ['A','B'] else ff['zero_C_h'];u=ff['teacher_'+v+'_u'] if v in ['A','B'] else ff['zero_C_u'];pp.append(np.exp(lp(h,u,mask,late,rF,S)))
   wanted['full_direct_AB']=(pp[0]+pp[1])/2;wanted['full_direct_ABC']=sum(pp)/3
   for v in ['F','C']:
    wanted['double_zero_'+v]=np.exp(lp(ff['zero_'+v+'_h'],ff['zero_'+v+'_u'],mask,late,rF,S));precision_changes+=int(wanted['double_zero_'+v].argmax()!=op['mean_context' if v=='F' else 'zero_C'].argmax())
   for k,want in wanted.items():err=float(abs(want-po[k]).max());mx=max(mx,err);assert err<1e-10 and want.argmax()==po[k].argmax();bridges+=1
   assert rec['choices_fixed_before_C_or_final_scores_used'] is True and rec['source_GT_used'] is False
  print('AUDIT126',key,n,flush=True)
 assert n==2304 and aliases==154 and choices==10750 and bridges==10750;out={'state':'PASS_NUMPY64_FULL_TEACHER_CHOICES_READOUTS_AND_UNCHANGED_ACTUAL_UPDATES','n':n,'aliases':aliases,'choice_checks':choices,'direct_readouts':bridges,'max_probability_error':mx,'max_objective_error':lossmx,'double_vs_GPU_zero_Top1_changes':precision_changes,'choice_counts':counts,'no_new_gradients':True,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
