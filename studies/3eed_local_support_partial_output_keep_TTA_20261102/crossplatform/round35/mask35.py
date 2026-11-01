"""Pure NumPy partial output-keep mask. No model, annotations, or media."""
import hashlib
import numpy as np

def make_masks(A,feedback,K=256):
 ids=[r['id'] for r in A];assert len(ids)==len(set(ids));lookup={rid:i for i,rid in enumerate(ids)}
 assert all(f['id_PRIVATE'] in lookup for f in feedback)
 fids={f['id_PRIVATE'] for f in feedback};U=np.array([i for i,rid in enumerate(ids) if rid not in fids],dtype=np.int64)
 old=np.zeros((len(A),K),bool);old[U]=True;partial=np.ones_like(old);union={rid:set() for rid in fids}
 for f in feedback:
  pos=list(f['positive_group_PRIVATE']);neg=list(f['negative_group_PRIVATE'])
  assert pos and neg and set(pos).isdisjoint(neg)
  assert all(isinstance(q,(int,np.integer)) and 0<=q<K for q in pos+neg)
  union[f['id_PRIVATE']].update(pos+neg)
 for rid,qs in union.items():partial[lookup[rid],sorted(qs)]=False
 assert np.all(partial[old]);assert int((partial&old).sum())==int(old.sum())
 detail=[]
 for i,r in enumerate(A):
  detail.append({'id_PRIVATE':r['id'],'alias':r.get('alias'),'feedback_observation':r['id'] in fids,'accepted_union_query_count':len(union.get(r['id'],set())),'retained_query_count':int(partial[i].sum()),'accepted_union_queries_PRIVATE':sorted(union.get(r['id'],set()))})
 count={'A_observations':len(A),'K':K,'feedback_observations':len(fids),'preferences':len(feedback),'original_U_observations':len(U),'original_U_coordinates':int(old.sum()),'partial_keep_coordinates':int(partial.sum()),'accepted_union_coordinates_freed':int((~partial).sum()),'partial_overlap_with_original_U_coordinates':int((partial&old).sum()),'new_keep_coordinates_from_feedback_rows':int((partial&~old).sum()),'old_mask_sha256':hashlib.sha256(old.tobytes()).hexdigest(),'partial_mask_sha256':hashlib.sha256(partial.tobytes()).hexdigest(),'normalization':'mean over retained scalar query-relative coordinates, including zero source-q0 anchor coordinates when retained'}
 assert count['partial_keep_coordinates']==len(A)*K-count['accepted_union_coordinates_freed']
 return U,old,partial,count,detail
