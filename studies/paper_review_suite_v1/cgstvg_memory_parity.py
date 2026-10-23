"""Bounded synthetic CPU/GPU parity for the exact attention allocation repair."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast
import json
from pathlib import Path
import sys
import time
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import atomic,sha
from cgstvg_memory import VENDOR,attention_source
D=R/'outputs/paper_review_suite_v1'


def main():
    import torch
    from torch import nn
    from timm.models.layers import trunc_normal_
    from stvg.runtime import setup,lease
    plan=json.loads((D/'cgstvg_memory_parity_plan.json').read_text());assert plan['code_sha256']==sha(Path(__file__))and plan['allocation_code_sha256']==sha(Path(__file__).with_name('cgstvg_memory.py'))
    assert not(D/'cgstvg_memory_parity_receipt.json').exists()
    setup();gpu=lease();torch.manual_seed(20261024);started=time.perf_counter();records=[]
    try:
        path=VENDOR/'models/vidswin/video_swin_transformer.py';tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='WindowAttention3D');scope=dict(torch=torch,nn=nn,trunc_normal_=trunc_normal_)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
        original,modified=attention_source();other=dict(scope);exec(compile(modified,__file__,'exec'),other);patched=other['forward']
        with torch.no_grad():
            for device in ['cpu','cuda']:
                for dim,heads,batches in [(96,3,8),(192,6,8),(384,12,4),(768,24,4)]:
                    model=scope['WindowAttention3D'](dim,(8,7,7),heads,qkv_bias=True).eval().requires_grad_(False).to(device)
                    x=torch.randn(batches,392,dim,device=device)
                    for masked in [False,True]:
                        mask=torch.zeros(2,392,392,device=device)if masked else None
                        if mask is not None:mask[:,::3,::5]=-100
                        before=model(x,mask);after=patched(model,x,mask);assert torch.equal(before,after)
                        records.append(dict(device=device,dim=dim,heads=heads,windows=batches,masked=masked,bitwise=True))
                    del model,x,before,after,mask
                    if device=='cuda':torch.cuda.empty_cache()
            model=scope['WindowAttention3D'](96,(8,7,7),3,qkv_bias=True).eval().requires_grad_(False).cuda();x=torch.randn(512,392,96,device='cuda');peaks=[];outputs=[]
            for fn in [model.forward,lambda x:patched(model,x)]:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();y=fn(x);torch.cuda.synchronize();peaks.append(torch.cuda.max_memory_allocated());outputs.append(y.cpu());del y
            assert torch.equal(*outputs)and peaks[1]<peaks[0]
        result=dict(status='passed_exact_CPU_GPU_attention_parity',fixtures=records,large_fixture_windows=512,large_fixture_bitwise=True,peak_allocated_bytes=dict(original=peaks[0],reuse=peaks[1]),original_AST_reverse_edit_exact=True,no_resolution_precision_sampling_or_context_change=True,CG_checkpoint_forwards=0,expert_calls=0,GT_or_real_RGB_access=0,seconds=time.perf_counter()-started,plan_sha256=sha(D/'cgstvg_memory_parity_plan.json'))
        atomic(D/'cgstvg_memory_parity_receipt.json',result);atomic(R/'results/paper_review_suite_v1/cgstvg_memory_parity.json',result);print(json.dumps(result))
    finally:gpu.close()


if __name__=='__main__':main()
