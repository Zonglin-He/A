"""Independent source selection and teacher evidence reconstruction; no target GT."""
from common149 import *
import numpy as np
from scipy.special import logsumexp
from audit_source149 import numpy_base
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct
from common149 import R,H,B,rd,sha,dump,check,utc

def main():
 check();al=rd(H/'SOURCE_ALPHA_LOCK.json');sr=rd(H/'SOURCE_CALIBRATION_ROWS_PRIVATE.json');summary=[];ks=rd(R/'research/round145/SOURCE_KAPPAS_LOCK.json')['kappa']
 for src in ['waymo','drone','quad']:
  for alpha in [.5,1.,2.]:
   vals=[next(v['IoU'] for v in r['values'] if v['alpha']==alpha) for r in sr if r['source']==src and r['eligible']];v=float(np.mean(vals))*100;expect=next(x for x in al['summary'] if x['source']==src and x['alpha']==alpha);assert abs(v-expect['mean_teacher_C_IoU'])<1e-12;summary.append((src,alpha,v))
  v=max(v for s,a,v in summary if s==src);candidates=[a for s,a,y in summary if s==src and y>=v-1e-12];assert al['alpha'][src]==min(candidates,key=lambda x:(x!=1,abs(np.log(x)),x))
 source_index={(e['entry']['source'],e['entry']['row']['id']):e for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json')}
 for r in sr:
  if not r['eligible']:continue
  src,sid=r['source'],r['id_PRIVATE'];a=np.load(R/'crossplatform/round145/source'/src/(sid+'.npz'));gs=[np.flatnonzero(a['assignment']==g) for g in range(int(a['assignment'].max()+1))];C=np.zeros_like(a['S'])
  for tag in ['pA','pB']:
   v=np.log(np.maximum([a[tag][g].mean() for g in gs],1e-300))[a['S']];C[a['S']]|=v.max()-v<=ks[src]
  f=source_index[src,sid]['entry']['files']['label'];assert sha(R/f['path'])==f['sha256'];y=np.load(R/f['path'])['ious'];pred=np.load(B/'source_head'/src/(sid+'.npz'));ids=np.concatenate([gs[g] for g in np.flatnonzero(C)])
  for pr,alpha in zip(pred['p'],pred['alphas']):
   q=int(ids[pr[ids].argmax()]);v=next(x for x in r['values'] if x['alpha']==alpha);assert q==v['query_PRIVATE'] and float(y[q])==v['IoU']
 n=0;mp=0.;exact=0;seal=rd(B/'EVIDENCE_SEALED.json');rows=rd(H/'INPUT_REGISTRY_PRIVATE.json')
 for f,h in seal['files'].items():assert sha(B/f)==h
 for src in ['waymo','drone','quad']:
  W=dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz')))
  for e in [e for e in rows if e['source']==src]:
   sid,tgt=e['row']['id'],e['target'];name=src+'_to_'+tgt;p=B/'evidence_PRIVATE'/name/(sid+'.npz');ev=np.load(p);old=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/p.name);a=np.load(R/'crossplatform/round148/method'/name/'samples_PRIVATE'/p.name);ps=dict(zip(a['arm_names'],a['scores']));gs=[np.flatnonzero(ev['assignment']==g) for g in range(int(ev['assignment'].max()+1))]
   for k in ['assignment','S','C','eligible','permutation']:assert np.array_equal(ev[k],old[k])
   if bool(ev['eligible']):
    f=np.load(R/'crossplatform/round145/image'/name/'features_PRIVATE'/p.name);b=numpy_base({k:f['norm_Bimage_'+k] for k in ['h','u','late','mask']},W);assert abs(b-ev['teacher_base']).max()<3e-4;alpha=al['alpha'][src];z=alpha*np.log(np.maximum(old['pB'],1e-300))+(1-alpha)*b;pr=np.exp(z-logsumexp(z));err=float(abs(pr-ev['cal_B']).max());mp=max(mp,err);assert err<5e-5;tar=np.array([ev['cal_B'][g].max() for g in gs])[ev['C']];tar=old['target'].copy() if alpha==1 else tar/tar.sum();assert np.array_equal(tar,ev['target']) and np.array_equal(tar[ev['permutation']],ev['wrong_target'])
   else:assert np.array_equal(ev['cal_B'],old['pB'])
   if al['alpha'][src]==1:assert np.array_equal(ev['cal_B'],old['pB']) and np.array_equal(ev['target'],old['target']);exact+=1
   assert bool(ev['same_teacher'])==np.array_equal(ev['target'],old['target']);effective=bool(ev['eligible'] and np.max(abs(ev['target']-ev['wrong_target']),initial=0)>1e-12);assert effective==bool(ev['effective_wrong'])
   for key,zero in [('direct_A','norm_Aimage'),('direct_AB','norm_joint_zero')]:assert np.array_equal(ev[key],direct(ps[zero],gs,ev['S'],ev['C'],ev['target']) if bool(ev['eligible']) else ps['Frozen_A'])
   n+=1
 assert n==1798;dump(H/'PREPARATION_AUDIT.json',dict(status='PASS',n=n,original_R145_C_S_eligibility_exact=True,max_independent_teacher_probability_error=mp,alpha1_exact_rows=exact,target_GT_read=False));print('PREP149_AUDIT',n,mp,flush=True)
if __name__=='__main__':main()
