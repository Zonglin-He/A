"""Finish metadata of already materialized 1798 evidence arrays, no head/model replay."""
from common149 import *
import numpy as np,collections

def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');alpha=rd(H/'SOURCE_ALPHA_LOCK.json')['alpha'];rows=[];counts=collections.Counter();eng=[];out=B/'evidence_PRIVATE';assert len(list(out.rglob('*.npz')))==1798
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];ev=np.load(out/(src+'_to_'+tgt)/(sid+'.npz'));old=np.load(R/'crossplatform/round145/evidence_PRIVATE'/(src+'_to_'+tgt)/(sid+'.npz'));eligible=bool(ev['eligible']);same=bool(ev['same_teacher']);changed=bool(eligible and ev['target'].argmax()!=old['target'].argmax());counts['n']+=1;counts['eligible']+=eligible;counts['changed_teacher']+=eligible and not same;counts['teacher_top_group_changed']+=changed;counts['new_plain_A_fits']+=eligible;counts['new_cal_A_fits']+=eligible and not same;counts['new_cal_AB_fits']+=eligible and not same;rows.append(dict(source=src,target=tgt,id_PRIVATE=sid,eligible=eligible,source_alpha=alpha[src],same_teacher=same,teacher_top_group_changed=changed,C_groups=int(ev['C'].sum()),S_groups=int(ev['S'].sum())))
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  rr=[e for e in reg if (e['source'],e['target'])==(src,tgt)]
  for active in [True,False]:
   for e in rr:
    ev=np.load(out/(src+'_to_'+tgt)/(e['row']['id']+'.npz'))
    if bool(ev['eligible'])==active:eng.append(e);break
 dump(H/'PREPARATION.json',{k:int(v) for k,v in counts.items()}|dict(GT_read=False,source_alpha=alpha,source_calibration_sha256=sha(H/'SOURCE_ALPHA_LOCK.json'),metadata_serialization_repair_only=True));dump(H/'EVIDENCE_ROWS_PRIVATE.json',rows);dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(B/'EVIDENCE_SEALED.json',dict(n=len(reg),GT_read=False,files={str(p.relative_to(B)):sha(p) for p in out.rglob('*.npz')}));dump(H/'PREPARATION_SERIALIZATION_REPAIR.json',dict(failure='numpyint counter could not serialize metadata after all1798 arrays written',prediction_arrays_not_changed=True,head_not_rerun=True,all_arrays_independently_audited_before_gradients=True,attempt_preserved=True));print('PREP149_FINALIZED',dict(counts),flush=True)
if __name__=='__main__':main()
