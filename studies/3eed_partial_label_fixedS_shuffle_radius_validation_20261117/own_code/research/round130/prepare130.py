"""Freeze a within-S label-identity intervention; no model or task labels."""
from pathlib import Path
import hashlib,json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round130';B=R/'crossplatform/round130';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def shuffle(S,C,assignment,salt):
 rng=np.random.default_rng(int(hashlib.sha256(('R130|fixedS|'+salt).encode()).hexdigest()[:16],16));sizes=np.bincount(assignment);wrong=C.copy();pairs=[]
 for n in sorted(set(sizes[S])):
  pos=rng.permutation(np.flatnonzero(S&C&(sizes==n)));neg=rng.permutation(np.flatnonzero(S&~C&(sizes==n)));m=min(len(pos),len(neg))
  for a,b in zip(pos[:m],neg[:m]):wrong[a]=False;wrong[b]=True;pairs.append([int(a),int(b)])
 assert not (wrong&~S).any() and wrong.sum()==C.sum() and wrong[assignment].sum()==C[assignment].sum()
 return wrong,pairs
def main():
 reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');rows=[];oldseal=rd(R/'crossplatform/round129/MATRIX_SEALED.json')
 for combo,h in oldseal['seals'].items():assert sha(R/'crossplatform/round129/formal'/combo/'SEALED.json')==h
 for e in reg:
  src,tgt,co,sid=e['source'],e['target'],e['cohort'],e['row']['id'];old=R/'crossplatform/round129/scope_PRIVATE'/co/(src+'_to_'+tgt)/(sid+'.npz');record={'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'protected_alias_fallback':e['protected_alias_fallback'],'partial_eligible':False,'effective_shuffle':False}
  if not e['protected_alias_fallback']:
   with np.load(old) as a:S,C,assignment=a['S'],a['C'],a['assignment'];wrong,pairs=shuffle(S,C,assignment,src+'|'+sid)
   meta=rd(old.with_suffix('.json'));active=meta['partial_eligible'];q0=int(np.load(old)['pF'].argmax());g0=assignment[q0];p=B/'control_PRIVATE'/co/(src+'_to_'+tgt)/(sid+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,S=S,C=C,wrong_C=wrong,assignment=assignment,pairs=np.array(pairs,dtype=np.int64).reshape(-1,2));record.update(partial_eligible=active,effective_shuffle=bool(pairs) and active,changed_groups=int((C!=wrong).sum()),same_supported_set=True,group_C_count=int(C.sum()),query_C_count=int(C[assignment].sum()),q0_in_S=bool(S[g0]),q0_in_C=bool(C[g0]),q0_in_wrong_C=bool(wrong[g0]),scope_groups=int(S.sum()),same_labels_no_effective_permutation=bool(active and not pairs),R129_scope_sha256=sha(old))
  rows.append(record)
 summary={'n':len(rows),'partial_eligible':sum(r['partial_eligible'] for r in rows),'effective_shuffle':sum(r['effective_shuffle'] for r in rows),'active_identity_controls':sum(r.get('same_labels_no_effective_permutation',False) for r in rows),'protected_aliases':sum(r['protected_alias_fallback'] for r in rows),'source_calibration_sha256':sha(R/'research/round129/SOURCE_CALIBRATION.json'),'R129_matrix_seal_sha256':sha(R/'crossplatform/round129/MATRIX_SEALED.json'),'C_observation_decoded':False,'GT_used':False,'shuffle_rule':'Within S and each equal query-count group-size bin, swap random disjoint retained/excluded groups up to maximum feasible swaps; identity if impossible, never label it effective. No GT or q0 used in construction.'}
 assert summary['n']==2304 and summary['partial_eligible']==257;dump(H/'CONTROL_ROWS_PRIVATE.json',rows);dump(H/'CONTROL_SUMMARY.json',summary);dump(H/'CONTROL_SEALED.json',dict(summary,files={str(p.relative_to(B)):sha(p) for p in (B/'control_PRIVATE').rglob('*.npz')}));print(summary)
if __name__=='__main__':main()
