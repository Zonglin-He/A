from common import *
import numpy as np
rows=[r for r in rd(H/'MECHANISM_ROWS_PRIVATE.json') if r['tta_A_update_reason']=='all_proposals_rejected'];same=0;delta_zero=0
for r in rows:
 p=B/(r['source']+'_to_'+r['target'])/'states_PRIVATE'/(r['id_PRIVATE']+'.npz')
 with np.load(p) as f:
  assert np.array_equal(f['tta_A_gradient1'],f['tta_A_gradient2']) and np.array_equal(f['tta_A_gradient1'],f['tta_A_gradient3']);same+=1
  assert not np.any(f['tta_A_delta']);delta_zero+=1
assert len(rows)==same==delta_zero==74;dump(H/'REJECTION_VECTOR_AUDIT.json',{'status':'PASS','n':74,'all3_gradient_vectors_bitexact_same':same,'final_delta_exact_zero':delta_zero,'GPU_or_updates':False,'FINAL_access':False});print('REJECTION_VECTOR_AUDIT_PASS',same)
