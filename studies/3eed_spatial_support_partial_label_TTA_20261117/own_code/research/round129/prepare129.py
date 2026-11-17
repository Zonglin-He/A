"""Freeze spatial qualification and teacher sets before target adaptation.

Only an exact copy of the existing frozen matching projection is replayed on
saved A/B/FAB activations. No point backbone, C arrays, raw media or task labels.
"""
from pathlib import Path
import json,hashlib,time,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from support129 import geometry,sets,wrong_sets
R=Path(__file__).resolve().parents[2];H=R/'research/round129';B=R/'crossplatform/round129';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2)+'\n')
def prob(head,h,u,mask,late,res):
 t=lambda x:torch.as_tensor(x,device='cuda');q=F.normalize(head(t(h)[None])[0]+t(late),dim=-1);s=q@t(u).T/.07;m=t(mask).bool();logits=s[:,m].logsumexp(-1)-s[:,~m].logsumexp(-1);return (logits.double()+t(res).double()).softmax(0).cpu().numpy()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);cal=rd(H/'SOURCE_CALIBRATION.json');reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');records=[];parity=0.;started=time.perf_counter()
 for src in ['waymo','drone','quad']:
  model=nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64)).cuda().eval().requires_grad_(False)
  with np.load(R/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as f:model.load_state_dict({k[6:]:torch.as_tensor(f[k].copy(),device='cuda') for k in f.files if k.startswith('match.')})
  for e in [x for x in reg if x['source']==src]:
   name=src+'_to_'+e['target'];sid=e['row']['id'];dest=B/'scope_PRIVATE'/e['cohort']/name/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if e['protected_alias_fallback']:records.append({'cohort':e['cohort'],'source':src,'target':e['target'],'id_PRIVATE':sid,'protected_alias':True,'scope_eligible':False,'partial_eligible':False});continue
   with np.load(R/e['prefix']) as f:xyz=f['xyz'][0];boxes=f['boxes']
   with np.load(R/e['observations']) as f:
    pre='real_B_' if e['cohort']=='DEV' else 'B_';present=pre+'valid' in f;valid=f[pre+'valid'] if present else np.zeros(1024,bool)
   geo=geometry(xyz,boxes,valid);oldroot=R/f"crossplatform/round127/formal/{e['cohort']}/{name}";metaold=rd(oldroot/'records_PRIVATE'/(sid+'.json'))
   with np.load(oldroot/'features_PRIVATE'/(sid+'.npz')) as f:
    late,res,mask=f['fixed_late'],f['fixed_r'],f['mask']
    with torch.no_grad():pa=prob(model,f['teacher_A_h'],f['teacher_A_u'],mask,late,res);pb=prob(model,f['teacher_B_h'],f['teacher_B_u'],mask,late,res);pF=prob(model,f['zero_F_h'],f['zero_F_u'],mask,late,res)
    err=float(abs((pa+pb)/2-f['teacher_full']).max());parity=max(parity,err);assert err<1e-11,err
   S,C,T,diag=sets(pa,pb,geo,cal);ws,wc,perm=wrong_sets(geo['groups'],S,C,src+'|'+sid);scopeok=bool(present and S.sum()>=2);partialok=scopeok and diag['has_partial_information'];q0=int(pF.argmax());qgroup=geo['assignment'][q0];meta={'cohort':e['cohort'],'source':src,'target':e['target'],'id_PRIVATE':sid,'protected_alias':False,'B_available':present,'old_R127_global_active':metaold['fit']['active'],'scope_eligible':scopeok,'partial_eligible':partialok,'groups':len(geo['groups']),'duplicate_queries':256-len(geo['groups']),'valid_unique_seed_count':geo['valid_unique_seed_count'],'unique_seed_count':geo['unique_seed_count'],'q0_in_S':bool(S[qgroup]),'q0_in_C':bool(C[qgroup]),'scope_changed_by_permutation':int((S!=ws).sum()),'labels_changed_by_permutation':int((C!=wc).sum()),'teacher_projection_error':err,'reference_relation_verified':'UNKNOWN; no geometric relation truth asserted','supported_locations_not_independent_returns':True,**diag}
   np.savez_compressed(dest,assignment=geo['assignment'],S=S,C=C,wrong_S=ws,wrong_C=wc,group_permutation=perm,n_A=geo['n_A'],n_B=geo['n_B'],coverage_B=geo['coverage_B'],pA=pa,pB=pb,pF=pF,teacher_global=(pa+pb)/2,teacher_conditional=T);dump(dest.with_suffix('.json'),meta);records.append(meta)
   if len(records)%128==0:print('SCOPE129',len(records),flush=True)
  del model;torch.cuda.empty_cache()
 dump(H/'SCOPE_ROWS_PRIVATE.json',records);dump(H/'SCOPE_SEALED.json',{'n':len(records),'scope_eligible':sum(r['scope_eligible'] for r in records),'partial_eligible':sum(r['partial_eligible'] for r in records),'teacher_projection_max_error':parity,'target_GT_used':False,'C_decoded':False,'seconds':time.perf_counter()-started,'files':{str(p.relative_to(B)):sha(p) for p in (B/'scope_PRIVATE').rglob('*') if p.is_file()}});print('SCOPE129_COMPLETE',len(records),sum(r['partial_eligible'] for r in records),parity,flush=True)
if __name__=='__main__':main()
