"""Numerical packed-Q/K parametrization lifecycle check; no data/model I/O."""
import json,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils import parametrize
from adapt54 import PackedQK
def digest(state,names):
 h=hashlib.sha256()
 for name in names:h.update(name.encode());h.update(state[name].detach().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 torch.manual_seed(54);m=nn.MultiheadAttention(288,8);before={k:v.detach().clone() for k,v in m.state_dict().items()};order=list(before);old=digest(before,order)
 parametrize.register_parametrization(m,'in_proj_weight',PackedQK(m.in_proj_weight));parametrize.remove_parametrizations(m,'in_proj_weight',leave_parametrized=False);after=m.state_dict();assert set(before)==set(after) and all(torch.equal(before[k],after[k]) for k in before)
 result={'state':'PASS_PACKED_QK_VALUES_PRESERVED_STATE_ORDER_DIFFERS','every_source_tensor_bitexact':True,'state_key_order_changed':list(after)!=order,'natural_order_digest_changed':digest(after,list(after))!=old,'original_order_digest_preserved':digest(after,order)==old,'data_or_model_checkpoint_reads':0,'GPU_work':0}
 p=Path(__file__).parent/'INTERFACE_LIFECYCLE_AUDIT.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
