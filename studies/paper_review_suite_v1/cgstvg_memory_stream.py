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
    modified='def forward(self, x, mask=None):\n    B_, N, C = x.shape\n    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)\n    q, k, v = qkv[0], qkv[1], qkv[2]\n    q = q * self.scale\n    relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)\n    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()\n    parts = []\n    for lo in range(0, B_, 64):\n        hi = min(lo + 64, B_)\n        attn = q[lo:hi] @ k[lo:hi].transpose(-2, -1)\n        attn.add_(relative_position_bias.unsqueeze(0))\n        if mask is not None:\n            indices = torch.arange(lo, hi, device=x.device) % mask.shape[0]\n            attn.add_(mask.index_select(0, indices).unsqueeze(1))\n        attn = torch.softmax(attn, dim=self.softmax.dim, out=attn)\n        attn = self.attn_drop(attn)\n        parts.append((attn @ v[lo:hi]).transpose(1, 2).reshape(hi - lo, N, C))\n    x = torch.cat(parts, dim=0)\n    x = self.proj(x)\n    x = self.proj_drop(x)\n    return x\n'
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
    try:yield dict(scope='Swin WindowAttention3D eval-only',operations='Same complete per-window FP32 attention;64independent window rows per batch; original global mask-row order and full-shape QKV/output projections',original_function_AST_recoverable=True,video_or_expert_context_chunking=False,independent_window_batch_size=64)
    finally:cls.forward=old
