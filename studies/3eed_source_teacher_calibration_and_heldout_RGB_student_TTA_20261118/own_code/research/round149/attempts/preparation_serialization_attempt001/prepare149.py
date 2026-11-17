"""Canonical R145 C and eligibility; only teacher identity and student RGB bank differ."""
from head149 import *
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct
from common149 import R,H,B,rd,sha,dump,check,utc
import collections

def main():
 check();torch.set_num_threads(4);alpha=rd(H/'SOURCE_ALPHA_LOCK.json')['alpha'];out=B/'evidence_PRIVATE';out.mkdir(exist_ok=False);reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');psseal=rd(R/'crossplatform/round148/method/SEALED.json');images=rd(R/'crossplatform/round145/image/SEALED.json');old_evseal=rd(R/'crossplatform/round145/EVIDENCE_SEALED.json');counts=collections.Counter();rows=[];eng=[]
 for src in ['waymo','drone','quad']:
  W={k:torch.from_numpy(v.copy()) for k,v in dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))).items() if k.startswith('match.')}
  for e in [e for e in reg if e['source']==src]:
   sid,tgt=e['row']['id'],e['target'];name=src+'_to_'+tgt;rel=name+'/samples_PRIVATE/'+sid+'.npz';assert sha(R/'crossplatform/round148/method'/rel)==psseal['files'][rel];a=np.load(R/'crossplatform/round148/method'/rel);ps=dict(zip(a['arm_names'],a['scores']));p=R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz');assert sha(p)==old_evseal['files'][str(p.relative_to(R/'crossplatform/round145'))];old=np.load(p);ass,S,C=old['assignment'],old['S'],old['C'];gs=groups(ass);eligible=bool(old['eligible']);pB=old['pB'];b=np.zeros(256);pn=pB.copy()
   if eligible:
    fpath=R/'crossplatform/round145/image'/name/'features_PRIVATE'/(sid+'.npz');assert sha(fpath)==images['files'][str(fpath.relative_to(R/'crossplatform/round145/image'))];ff=np.load(fpath);b=base_logits({k:ff['norm_Bimage_'+k] for k in ['h','u','late','mask']},W);pn=calibrated(pB,b,alpha[src]);t=np.array([pn[g].max() for g in gs])[C];t=old['target'].copy() if alpha[src]==1 else t/t.sum();perm=old['permutation'];wrong=t[perm]
   else:t=old['target'];wrong=old['wrong_target'];perm=old['permutation']
   same=bool(np.array_equal(t,old['target']));effective=bool(eligible and np.max(abs(t-wrong),initial=0)>1e-12);dd=out/name;dd.mkdir(exist_ok=True);vals={k:old[k] for k in ['assignment','S','C','eligible']};vals.update(pA=old['pA'],pB=pB,p0=old['p0'],teacher_base=b,cal_B=pn,plain_target=old['target'],plain_wrong_target=old['wrong_target'],target=t,wrong_target=wrong,permutation=perm,effective_wrong=np.array(effective),plain_effective_wrong=old['effective_wrong'],same_teacher=np.array(same),direct_A=direct(ps['norm_Aimage'],gs,S,C,t) if eligible else ps['Frozen_A'],direct_AB=direct(ps['norm_joint_zero'],gs,S,C,t) if eligible else ps['Frozen_A']);np.savez_compressed(dd/(sid+'.npz'),**vals);counts['n']+=1;counts['eligible']+=eligible;counts['changed_teacher']+=eligible and not same;counts['teacher_top_group_changed']+=eligible and t.argmax()!=old['target'].argmax();counts['new_plain_A_fits']+=eligible;counts['new_cal_A_fits']+=eligible and not same;counts['new_cal_AB_fits']+=eligible and not same;rows.append(dict(source=src,target=tgt,id_PRIVATE=sid,eligible=eligible,source_alpha=alpha[src],same_teacher=same,teacher_top_group_changed=bool(eligible and t.argmax()!=old['target'].argmax()),C_groups=int(C.sum()),S_groups=int(S.sum())))
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  rr=[e for e in reg if (e['source'],e['target'])==(src,tgt)]
  for kind in ['changed','sameB','noB']:
   for e in rr:
    ev=np.load(out/(src+'_to_'+tgt)/(e['row']['id']+'.npz'));ok=(bool(ev['eligible']) and not bool(ev['same_teacher'])) if kind=='changed' else (bool(ev['eligible']) and bool(ev['same_teacher'])) if kind=='sameB' else not e['has_B']
    if ok:
     if e not in eng:eng.append(e)
     break
 dump(H/'PREPARATION.json',dict(counts)|dict(GT_read=False,source_alpha=alpha,source_calibration_sha256=sha(H/'SOURCE_ALPHA_LOCK.json')));dump(H/'EVIDENCE_ROWS_PRIVATE.json',rows);dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(B/'EVIDENCE_SEALED.json',dict(n=len(reg),GT_read=False,files={str(p.relative_to(B)):sha(p) for p in out.rglob('*.npz')}));print('PREP149',dict(counts),flush=True)
if __name__=='__main__':main()
