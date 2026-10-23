"""Scoped eval-only allocation changes, preserving native attention equations."""
import ast
import inspect
from contextlib import contextmanager
from pathlib import Path

R=Path(__file__).resolve().parents[2]
VENDOR=R.parent/'outputs/maintenance/engineering_speed_v1/baseline_workspace/external/CGSTVG'


def attention_source():
    tree=ast.parse((VENDOR/'models/vidswin/video_swin_transformer.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='WindowAttention3D')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='forward')
    original=ast.unparse(fn)
    changes=[
        ('attn = attn + relative_position_bias.unsqueeze(0)','attn.add_(relative_position_bias.unsqueeze(0))'),
        ('attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)','attn = attn.view(B_ // nW, nW, self.num_heads, N, N)\n        attn.add_(mask.unsqueeze(1).unsqueeze(0))'),
        ('attn = self.softmax(attn)','attn = torch.softmax(attn, dim=self.softmax.dim, out=attn)'),
    ]
    modified=original
    for before,after in changes:
        assert modified.count(before)==(2 if 'self.softmax' in before else 1)
        modified=modified.replace(before,after)
    restored=modified
    for before,after in reversed(changes):restored=restored.replace(after,before)
    assert restored==original
    return original,modified


@contextmanager
def inference_allocation(model):
    import torch
    assert not model.training and not torch.is_grad_enabled()
    modules=[x for x in model.vid.modules()if type(x).__name__=='WindowAttention3D']
    assert modules and len({type(x)for x in modules})==1
    cls=type(modules[0]);old=cls.forward;original,modified=attention_source()
    assert ast.dump(ast.parse(inspect.getsource(old).strip()))==ast.dump(ast.parse(original))
    ns=dict(old.__globals__);exec(compile(modified,__file__,'exec'),ns)
    cls.forward=ns['forward']
    try:yield dict(scope='Swin WindowAttention3D eval-only',operations='same FP32 relative-bias addition, mask addition and softmax with reused output storage',original_function_AST_recoverable=True,temporal_or_window_chunking=False)
    finally:cls.forward=old
