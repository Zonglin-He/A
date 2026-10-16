"""Physical merged-frame timestamps in native mRoPE, preserving all other indexing."""
import hashlib,inspect,textwrap,types,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def corrected(original):
    src=textwrap.dedent(inspect.getsource(original))
    dtype='dtype=range_tensor.dtype, device=range_tensor.device';assert src.count(dtype)==1
    src=src.replace(dtype,'dtype=torch.float64, device=range_tensor.device')
    needle='time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second'
    assert src.count(needle)==1
    line=next(x for x in src.splitlines() if needle in x);indent=line[:len(line)-len(line.lstrip())]
    replacement='physical = torch.as_tensor(self._stvg_merged_elapsed[video_index - 1], dtype=torch.float64, device=expanded_range.device).view(-1, 1)\n'+indent+'assert len(physical) == llm_grid_t\n'+indent+'time_tensor = physical.expand(-1, llm_grid_h * llm_grid_w) * self.config.vision_config.tokens_per_second'
    fixed=src.replace(needle,replacement);ns=dict(original.__globals__);exec(compile(fixed,'<stvg_physical_merged_clock_v1>','exec'),ns)
    return ns[original.__name__],dict(status='qualified_source_patch',source_sha256=hashlib.sha256(fixed.encode()).hexdigest(),scope='explicit physical merged-frame elapsed; native text/spatial offsets and suffix/delta construction unchanged')
def apply(model):
    fn,proof=corrected(type(model.model).get_rope_index);model.model.get_rope_index=types.MethodType(fn,model.model);return proof
def bind_job(model,job):
    import math
    elapsed=[]
    for v in job['videos']:
        ts=v['actual_timestamps_seconds'];assert len(ts)>=2 and all(math.isfinite(x) for x in ts) and all(b>a for a,b in zip(ts,ts[1:]))
        elapsed.append([float(ts[i]-ts[0]) for i in range(0,len(ts),2)])
    model.model._stvg_merged_elapsed=elapsed;return elapsed
def verify_positions(model,job,pos,runs):
    import torch
    elapsed=bind_job(model,job);rate=model.config.vision_config.tokens_per_second
    for e,run in zip(elapsed,runs):
        assert len(run)==len(e)*256
        observed=pos[0,0,run].reshape(len(e),256);base=observed[0,0]
        expected=(torch.tensor(e,dtype=torch.float64,device=observed.device)*rate).long()+base
        assert torch.equal(observed,expected[:,None].expand_as(observed))
    assert int(pos.max())<model.config.max_position_embeddings
    return dict(status='pass',actual_merged_elapsed=elapsed,tokens_per_second=rate,physical_temporal_coordinates_exact=True,positions_maximum=int(pos.max()),native_spatial_and_text_construction=True)
def cpu_contracts():
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from scripts.stvg_qwen_uniform_clock002 import corrected_function
    original=Qwen2_5_VLModel.get_rope_index;fn,_=corrected(original);uniform,_=corrected_function(original)
    obj=types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2,tokens_per_second=2),image_token_id=98,video_token_id=99,vision_start_token_id=97))
    count=0
    for times in ([0.,.5,1.],[0.,.45,1.25],[0.,.001,4.1]):
        obj._stvg_merged_elapsed=[times];ids=torch.tensor([[1,2,97]+[99]*12+[3,4]]);grid=torch.tensor([[3,4,4]]);mask=torch.ones_like(ids)
        a,d=fn(obj,ids,video_grid_thw=grid,second_per_grid_ts=[.5],attention_mask=mask)
        expected=(torch.tensor(times,dtype=torch.float64)*2).long()+3
        assert torch.equal(a[0,0,3:15].reshape(3,4)[:,0],expected) and d.item()==a.max().item()+1-ids.shape[1]
        if times==[0.,.5,1.]:
            b,e=uniform(obj,ids,video_grid_thw=grid,second_per_grid_ts=[.5],attention_mask=mask);assert torch.equal(a,b) and torch.equal(d,e)
        count+=1
    return dict(status='pass',count=count,CPU_only=True,GT_read=False,regular_uniform_exact=True,irregular_physical_formula_exact=True)
if __name__=='__main__':
    import json
    print(json.dumps(cpu_contracts()))
