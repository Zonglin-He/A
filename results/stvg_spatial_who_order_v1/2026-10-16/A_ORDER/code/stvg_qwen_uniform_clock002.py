"""Preserve fractional seconds in the installed Qwen uniform-video mRoPE.

An isolated per-model method binding changes one dtype coercion, not weights,
the processor, prompts, decoder or sampling. The original method stays intact.
"""
import inspect, textwrap, types, hashlib

def corrected_function(original):
    source=textwrap.dedent(inspect.getsource(original))
    needle='dtype=range_tensor.dtype, device=range_tensor.device'
    assert source.count(needle)==1
    fixed=source.replace(needle,'dtype=torch.float64, device=range_tensor.device')
    namespace=dict(original.__globals__)
    exec(compile(fixed,'<stvg_fractional_uniform_clock002>','exec'),namespace)
    return namespace[original.__name__],dict(original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),corrected_function_sha256=hashlib.sha256(fixed.encode()).hexdigest(),scope='single uniform-grid seconds dtype coercion; no other AST changes')

def apply(model):
    original=type(model.model).get_rope_index
    fn,proof=corrected_function(original)
    model.model.get_rope_index=types.MethodType(fn,model.model)
    return proof

def contracts():
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    c=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2,tokens_per_second=25),image_token_id=98,video_token_id=99,vision_start_token_id=97)
    obj=types.SimpleNamespace(config=c);orig=Qwen2_5_VLModel.get_rope_index;fixed,proof=corrected_function(orig);count=0
    for t in (4,8,16):
        # Three prefix tokens followed by video patch tokens and text suffix.
        ids=torch.tensor([[1,2,97]+[99]*(t*4)+[3,4]])
        grid=torch.tensor([[t,4,4]]);mask=torch.ones_like(ids)
        for seconds in (.2,.8,1.,2.):
            a,d=fixed(obj,ids,video_grid_thw=grid,second_per_grid_ts=[seconds],attention_mask=mask)
            expected=(torch.arange(t,dtype=torch.float64)*seconds*25).long()+3
            assert torch.equal(a[0,0,3:3+t*4].reshape(t,4)[:,0],expected)
            assert d.item()==a.max().item()+1-ids.shape[1]
            if seconds.is_integer():
                b,e=orig(obj,ids,video_grid_thw=grid,second_per_grid_ts=[seconds],attention_mask=mask)
                assert torch.equal(a,b)and torch.equal(d,e)
            count+=1
    return dict(status='pass',count=count,CPU_only=True,GT_read=False,proof=proof,integer_clock_exact=True,fractional_clock_formula_exact=True)

if __name__=='__main__':
    import json
    print(json.dumps(contracts()))
