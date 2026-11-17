"""Per-own-source finite teacher-only scaling selection after sealed predictions."""
from head149 import *
import collections

def main():
 check();assert rd(H/'SOURCE_HEAD_AUDIT.json')['status']=='PASS';ks=rd(R/'research/round145/SOURCE_KAPPAS_LOCK.json')['kappa'];rows=[];sc=collections.defaultdict(list);counts=collections.Counter();root=B/'source_head';seal=rd(root/'SEALED.json')
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  en=e['entry'];src,sid=en['source'],en['row']['id'];p=root/src/(sid+'.npz');assert sha(p)==seal['files'][src+'/'+p.name];a=np.load(R/'crossplatform/round145/source'/src/p.name);has=rd(R/'crossplatform/round145/source'/src/(sid+'.json'))['has_B'];gs=groups(a['assignment']);C=original_C(a['pA'],a['pB'],gs,a['S'],ks[src],has);eligible=bool(has and a['S'].sum()>=2 and C.any());r=dict(source=src,id_PRIVATE=sid,eligible=eligible,values=[])
  if eligible:
   l=en['files']['label'];assert sha(R/l['path'])==l['sha256'];ious=np.load(R/l['path'])['ious'];out=np.load(p);counts[src]+=1
   for alpha,pr in zip(out['alphas'],out['p']):
    ids=np.concatenate([gs[i] for i in np.flatnonzero(C)]);q=int(ids[pr[ids].argmax()]);v=float(ious[q]);r['values'].append(dict(alpha=float(alpha),IoU=v,query_PRIVATE=q));sc[src,float(alpha)].append(v)
  rows.append(r)
 alphas=rd(H/'experiment/DESIGN_LOCK.json')['source_alpha_candidates'];summ=[dict(source=s,alpha=x,n=len(sc[s,x]),mean_teacher_C_IoU=float(np.mean(sc[s,x]))*100) for s in ['waymo','drone','quad'] for x in alphas];selected={}
 for s in ['waymo','drone','quad']:
  rr=[r for r in summ if r['source']==s];best=max(r['mean_teacher_C_IoU'] for r in rr);cand=[r['alpha'] for r in rr if r['mean_teacher_C_IoU']>=best-1e-12];selected[s]=min(cand,key=lambda x:(x!=1,abs(np.log(x)),x))
 dump(H/'SOURCE_CALIBRATION_ROWS_PRIVATE.json',rows);dump(H/'SOURCE_ALPHA_LOCK.json',dict(status='SOURCE_OWN_FINITE_CALIBRATION_LOCKED',alpha=selected,summary=summ,counts=dict(counts),source_labels_only=True,no_target_selection=True,source_weight_training=False,all_eligible_not_GT_prefiltered=True,source_seal_sha256=sha(root/'SEALED.json'),utc=utc()));print('SOURCE_ALPHA',selected,summ,flush=True)
if __name__=='__main__':main()
