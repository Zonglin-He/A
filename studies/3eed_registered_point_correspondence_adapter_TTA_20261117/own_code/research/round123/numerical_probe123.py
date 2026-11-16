"""Pre-score independent CPU kernel qualification on6 real engineering states."""
from audit123 import *
def main():
 rows=[]
 for src in ['waymo','drone','quad']:
  S=dict(np.load(B/(src+'_AUDIT_MODEL_PRIVATE.npz')));radius=float(S['original_adapter_radius'])
  for co in ['DEV','EXP']:
   D=next(p for p in (B/'engineering'/co).glob(src+'_to_*') if p.is_dir());rs=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');e=next(x for x in rs if x['cohort']==co and x['source']==src and x['target']==D.name.split('_to_')[1] and not x['protected_alias_fallback'] and (D/'states_PRIVATE'/(x['row']['id']+'.npz')).exists() and rd(D/'records_PRIVATE'/(x['row']['id']+'.json'))['fit']['active']);sid=e['row']['id'];z=np.load(R/e['prefix']);ob=np.load(R/e['observations']);pre='real_B_' if co=='DEV' else 'B_';A=torch.from_numpy(z['vis0'][0]);BB=torch.from_numpy(ob[pre+'vis']);ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));st=np.load(D/'states_PRIVATE'/(sid+'.npz'));ids=torch.from_numpy(ff['fit_ids']);far=torch.from_numpy(ff['fit_far']);perm=torch.from_numpy(ff['fit_permutation']);pred,get=reproduce(A,BB,ids,far,perm,S,radius,'R123|'+src+'|'+sid);er={k:float(abs(v-torch.from_numpy(st[k])).max()) for k,v in pred.items()};assert max(er.values())<5e-6;rows.append({'source':src,'cohort':co,'max_element_error':max(er.values()),'states':len(er)})
 out={'state':'PASS6_REAL_CPU_POINT_LOSS_AND_NORMALIZED_UPDATE_KERNEL_BEFORE_TASK_SCORING','n':6,'checks':rows,'GT_used':False};(H/'NUMERICAL_PROBE.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
