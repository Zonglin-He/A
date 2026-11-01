"""CPU synthetic contract checks and real ID/group mask counts. No fit or GT."""
import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round35')]
from mask35 import make_masks
from common35 import rows,read,dump,sha,BASE,EXP

def main():
 A=[{'id':x,'alias':x} for x in ['a','b','c']]
 # Repeated endpoint queries count once; two accepted pairs may share endpoints.
 ff=[{'id_PRIVATE':'a','positive_group_PRIVATE':[0,1],'negative_group_PRIVATE':[2]}, {'id_PRIVATE':'a','positive_group_PRIVATE':[1],'negative_group_PRIVATE':[3]}, {'id_PRIVATE':'b','positive_group_PRIVATE':[3],'negative_group_PRIVATE':[1]}]
 U,old,new,c,d=make_masks(A,ff,K=5)
 assert U.tolist()==[2] and old.sum()==5 and new.sum()==9 and (new&old).sum()==5
 assert np.flatnonzero(new[0]).tolist()==[4] and np.flatnonzero(new[1]).tolist()==[0,2,4]
 # One hypothetical q0 stays a source anchor. Mask does not create C negatives.
 rng=np.random.default_rng(71);source=rng.normal(size=(3,5));current=source+rng.normal(size=(3,5))*.03;q0=np.array([4,1,3]);idx=np.arange(3)
 delta=(current-current[idx,q0,None])-(source-source[idx,q0,None]);v=np.square(delta[new]).mean();manual=sum(delta[i,j]**2 for i in range(3) for j in range(5) if new[i,j])/int(new.sum());assert abs(v-manual)<1e-16
 shifted=current+rng.normal(size=(3,1));delta2=(shifted-shifted[idx,q0,None])-(source-source[idx,q0,None]);assert np.allclose(delta,delta2,atol=1e-14)
 assert np.array_equal(old,make_masks(A,ff,K=5)[1])
 # Derived-strength control isolates the added feedback-row coordinates.
 oldmatched=np.square(delta[old]).sum()/int(new.sum());added=np.square(delta[new&~old]).sum()/int(new.sum())
 assert abs(v-oldmatched-added)<1e-16
 assert abs(oldmatched-np.square(delta[old]).mean()*(old.sum()/new.sum()))<1e-16
 actual={};details=[];f=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(f)==22
 for pl in ['drone','quad']:
  AA=[r for r in rows() if r['platform']==pl];F=[x for x in f if x['platform']==pl];u,o,n,count,detail=make_masks(AA,F)
  assert len(u)=={'drone':96,'quad':93}[pl] and len(F)=={'drone':10,'quad':12}[pl]
  assert n.shape==(104,256) and np.all(n[u]);actual[pl]=count;details.append({'platform':pl,'rows_PRIVATE':detail})
 dump(EXP/'MASK_COUNTS_LOCK.json',actual);dump(EXP/'MASK_ROWS_PRIVATE.json',details)
 result={'state':'PASS','pure_CPU_no_torch_or_GPU':True,'no_task_labels_or_media_read':True,'duplicate_endpoints_union_not_recounted':True,'all_U_coordinates_retained':True,'normalization_by_actual_retained_count':True,'matched_denominator_control_exact_additive_isolation':True,'derived_U_strength_ratio_no_GT_or_tuning':True,'source_q0_relative_shift_invariant':True,'real_contract_counts':actual,'mask_program_sha256':sha(BASE/'mask35.py'),'no_fit_performed':True};dump(Path(__file__).resolve().parent/'CPU_CHECK_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
