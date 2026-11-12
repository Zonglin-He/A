"""CPU native attention/read routing mechanics; no source model/data labels."""
from core88 import *
import types
seed=88001;torch.manual_seed(seed);att=torch.nn.MultiheadAttention(288,4,dropout=0.).eval();q=torch.randn(5,1,288);kv=torch.randn(12,1,288)
with torch.no_grad():
 x,w=att(q,kv,kv);y,v=F.multi_head_attention_forward(q,kv,kv,att.embed_dim,att.num_heads,att.in_proj_weight,att.in_proj_bias,None,None,False,0.,att.out_proj.weight,att.out_proj.bias,training=False,need_weights=True);err=float(abs(x-y).max());assert err<1e-6
 mask=torch.zeros(5,12);mask[:,:6]=float(np.log(.05));z,_=att(q,kv,kv,attn_mask=mask);assert float(abs(x-z).max())>1e-3
 perm=torch.randperm(12);same,_=att(q,kv[perm],kv[perm],attn_mask=mask[:,perm]);assert float(abs(z-same).max())<1e-6
 wrong,_=att(q,kv,kv,attn_mask=mask[:,perm]);assert float(abs(z-wrong).max())>1e-3
assert not torch.cuda.is_initialized();dump(H88/'SYNTHETIC_AUDIT.json',{'state':'PASS_R88_NATIVE_FUNCTIONAL_QKV_PARITY_LOCAL_INTERVENTION_AND_PERMUTATION','native_parity_error':err,'correct_key_value_mask_permutation_equivariant':True,'wrong_mask_changes_actual_output':True,'new_GT_scoring':0});print('PASS')
