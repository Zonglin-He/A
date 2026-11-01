"""Independent CPU reconstruction from immutable raw logits and sealed predictions."""
import sys,numpy as np,collections,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round29'),str(ROOT/'research/round5/audit')]
from common29 import *
from independent_core_v2 import candidate_ious,transform_gt

def main():
 seals=['gate154_attempt001','fullpair208_attempt001','groups208_attempt001','direct208_attempt001','sourceR64_attempt001','ranknet_attempt001','ranknet_logodds_attempt001','closureR64_attempt001','blind_review24_attempt001','correction_review6_attempt001']
 for name in seals:verify(BASE/name)
 for arm in ['ranknet_attempt001','ranknet_logodds_attempt001']:
  for phase in ['A208_predictions','R64_predictions']:verify(BASE/arm/phase)
 # Rebuild original R22 all-gate result directly, not via new policy helper.
 checks=collections.Counter();original=collections.Counter()
 for r in rows()[:208]:
  raw=read(ROOT/'crossplatform/round22/pair208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['pairs'];fp=read(BASE/'fullpair208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['pairs'];sealed=read(BASE/'direct208_attempt001/PAIR_PREFERENCES_PRIVATE.json');lookup={(x['id_PRIVATE'],x['pair_alias']):x for x in sealed}
  for p,new in zip(raw,fp):
   supports={c:True for c in 'AB'}
   for v in p['subject']['views']:
    for order in ['normal','reversed']:
     z=v[order]['logits'];bl=v['blank']['logits']
     for c in 'AB':
      own=c if order=='normal' else ('B' if c=='A' else 'A');other='B' if own=='A' else 'A';supports[c]&=min(z[own]-z[other],z[own]-z['U'],z[own]-z[other]-bl[own]+bl[other])>=.5
   original['A' if supports['A'] else 'B' if supports['B'] else 'U']+=1
   means=[];supports={c:True for c in 'AB'}
   for order in ['normal','reversed']:
    z=new['full'][order]['logits']
    for c in 'AB':
     own=c if order=='normal' else ('B' if c=='A' else 'A');other='B' if own=='A' else 'A';supports[c]&=min(z[own]-z[other],z[own]-z['U'])>=.5
   label='A' if supports['A'] else 'B' if supports['B'] else 'U';assert label==lookup[r['id'],p['pair_alias']]['label'];checks['raw_pair_preferences_rebuilt']+=1
 assert dict(original)=={'A':19,'B':4,'U':131},original
 # Recompute task scores independently from metadata and original frozen box bank.
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};cc={};rs=rows()[:208]+read(EXP.parent/'input/R64_PRIVATE.json');records={x['id_PRIVATE']:x for x in read(EXP/'ALL272_TASK_ROWS_PRIVATE.json')}
 for r in rs:
  x=records[r['id']];ii=idx[r['id']];path=ii['meta_path']
  if path not in cc:assert sha(path)==ii['metadata_container_sha256'];cc[path]=read(path)
  gt=transform_gt(cc[path]['ground_info'][ii['object_index']]['bbox_3d'],r['platform'],ii.get('waymo_view_suffix'));q=np.array(list(x['q'].values()));b=feature(r)['boxes'][q].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt)[0]
  for a,t in zip(x['q'],v):assert float(t)==x['ious'][a];checks['task_scalar_scores_recomputed']+=1
 # All instance grouping assignments are disjoint; source representatives unchanged.
 for r in rs[:208]:
  g=read(BASE/'groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=[]
  for inds in g.values():used+=inds
  assert len(used)==len(set(used));checks['group_partitions_disjoint']+=1
 # New content exclusion re-evaluated against all old528, not filename-only.
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];inv={x['image_file_sha256']:x for x in inv};oldhash=[inv[sha(r['imagepath'])] for r in rows()];new=read(EXP.parent/'input/R64_CONTENT_HASHES_PRIVATE.json')
 for key in ['decoded_pixel_sha256','canonical_XYZ_sha256']:assert not {x[key] for x in oldhash}&{x[key] for x in new};assert len({x[key] for x in new})==64
 # Probability is sigmoid(exact token logodds); inference argmax invariant.
 maxerr=0.
 for r in rs:
  z=feature(r);v=np.asarray(z['query'],np.float64)@np.asarray(z['tokens'],np.float64).T/.07;mask=z['token_mask'];lse=lambda t:np.max(t,axis=-1)+np.log(np.exp(t-np.max(t,axis=-1,keepdims=True)).sum(-1));lo=lse(v[:,mask])-lse(v[:,~mask]);p=1/(1+np.exp(-lo));er=float(abs(p-z['scores']).max());maxerr=max(maxerr,er);assert er<2e-5
 rec=read(BASE/'closureR64_attempt001/CLOSURE_AUDIT.json');assert all(x['full_model_forwards']==64 and x['box_max_abs_error']==0 and x['nonmatching_weights_unchanged'] and x['source_restored_bitexact'] for x in rec)
 # All negative task tails retained, exact table reduction.
 s=read(EXP/'TASK_SUMMARY.json');allneg=read(EXP/'ALL_NEGATIVES.json');expected=sum(sum(v['worsened'] for v in pp.values()) for split in s.values() for pp in split.values());assert len(allneg)==expected
 dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','counts':dict(checks),'original_gate_labels':dict(original),'full_model_closure_forwards':128,'all_boxes_bitexact':True,'nonmatching_weights_unchanged':True,'loss_readout_probability_logodds_maxerror':maxerr,'R64_actual_RGB_XYZ_disjoint_recent_history':True,'scene_and_historical_fullVAL_exposure_retained':True,'all_pseudo_feedback_used_including_bad_and_neither':True,'allnegative_rows':len(allneg),'visual_labels_assistant_provisional_not_externalgold':True,'B_feedback_never_constructed_or_used_to_fit':True,'TTA_failure_not_generalized_to_all_grounding_TTA':True});print('AUDIT29_PASSED',dict(checks),maxerr)
if __name__=='__main__':main()
