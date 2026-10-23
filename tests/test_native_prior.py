"""Native-prior adapter and removal of misleading output aliases."""
import numpy as np
import torch
from stvg.prior import native_prior,from_capture
from stvg.temporal import mass_table,probabilities
from stvg.worker import prediction_arrays

def test_legacy_cache_uses_last_layer_only_and_recovers_native_exactly():
 ids=[0,1,4,5,8,9];records=[{'frame_ids':ids[::2]},{'frame_ids':ids[1::2]}]
 gen=torch.Generator().manual_seed(7);logs=[torch.randn(6,1,3,2,generator=gen)for _ in range(2)]
 q=from_capture(dict(records=records,logits=logs,frame_ids=ids))
 _,ref=mass_table(probabilities(records,[x[-1]for x in logs]));assert np.array_equal(q,ref)
 for x in logs:x[:-1].add_(1000)
 assert np.array_equal(from_capture(dict(records=records,logits=logs,frame_ids=ids)),q)
 assert np.array_equal(native_prior(records,[x[-1]for x in logs],ids),q)

def test_selected_output_is_never_mislabeled_Global6():
 boxes=np.ones((8,4));out=dict(Source=boxes,E_Ref=boxes,BoxFixed=boxes,positions8=list(range(8)),intervals={'Native':[0,8]},interval=[1,7])
 arr=prediction_arrays(out,list(range(8)))
 assert 'Global6_interval'not in arr and np.array_equal(arr['Selected_interval'],[1,7])
