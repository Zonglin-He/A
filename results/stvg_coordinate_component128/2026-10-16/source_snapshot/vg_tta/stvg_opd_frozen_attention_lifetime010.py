"""Original frozen attention arithmetic with storage reuse for ephemeral tensors and immediate dead-reference release."""
from contextlib import contextmanager
import inspect, sys, textwrap
import torch
from vg_tta.stvg_opd_frozen_module_cpu008 import frozen_forward as capture008


def storage_forward(original, namespace):
    source = textwrap.dedent(inspect.getsource(original))
    bias = 'attn = attn + relative_position_bias.unsqueeze(0)'
    mask = 'attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)'
    assert source.count(bias) == source.count(mask) == 1
    assert source.count('attn = self.softmax(attn)') == 2
    source = source.replace(bias,
        'attn = attn.to(torch.promote_types(attn.dtype, relative_position_bias.dtype))\n'
        '    attn.add_(relative_position_bias.unsqueeze(0))')
    source = source.replace(mask,
        'attn = attn.view(B_ // nW, nW, self.num_heads, N, N)\n'
        '        attn = attn.to(torch.promote_types(attn.dtype, mask.dtype))\n'
        '        attn.add_(mask.unsqueeze(1).unsqueeze(0))')
    product = 'attn = q @ k.transpose(-2, -1)'
    assert source.count(product) == 1
    source = source.replace(product, product + '\n    del q, k')
    aggregate = 'x = (attn @ v).transpose(1, 2).reshape(B_, N, C)'
    assert source.count(aggregate) == 1
    source = source.replace(aggregate, aggregate + '\n    del attn, qkv, v, relative_position_bias')
    source = source.replace('attn = self.softmax(attn)',
        'attn = torch.softmax(attn, dim=self.softmax.dim, out=attn)')
    # Storage reuse is only valid for the original frozen, eval-only prefix.
    signature = 'def forward(self, x, mask=None):'
    assert source.count(signature) == 1
    source = source.replace(signature, signature + '\n'
        '    assert not self.training and not torch.is_grad_enabled()\n')
    local = dict(namespace)
    exec(compile(source, __file__, 'exec'), local)
    return local['forward']


@contextmanager
def frozen_attention_storage(model):
    classes = {type(module) for module in model.vid.modules()
               if type(module).__name__ == 'WindowAttention3D'}
    assert classes
    saved = {cls: cls.forward for cls in classes}
    try:
        for cls, original in saved.items():
            cls.forward = storage_forward(original, vars(sys.modules[cls.__module__]))
        yield
    finally:
        for cls, original in saved.items():
            cls.forward = original


def frozen_forward(model, frames, row):
    with frozen_attention_storage(model):
        return capture008(model, frames, row)
