import torch
import pytest
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v3_native_scopes_v2 import configure_branch

@pytest.mark.parametrize('branch',['event','spatial'])
def test_scopes_include_real_query_and_temporal_modules_and_exclude_other_branch(branch):
    torch.manual_seed(29)
    model=Desta3DAdapterV2(in_channels=8,query_dim=8,hidden_dim=128,architecture='dual3d')
    prior=set()
    for level in range(4):
        named=configure_branch(model,branch,level);names={n for n,p in named}
        assert prior<=names;prior=names
        assert not any(n.startswith('norm_stem.') for n in names)
        if level<3:assert not any(n.startswith(('input_proj.','shared_stem.')) for n in names)
        if level>=2:
            assert any(n.startswith(f'query_pool_{branch}.') for n in names)
            if branch=='event':assert {'event_temporal_reader.paths.0.1.weight','event_temporal_reader.paths.0.1.bias'}<=names
        other='spatial' if branch=='event' else 'event'
        assert not any(n.startswith((f'query_pool_{other}.',f'film_{other}.',f'norm_{other}.',f'{other}_reader.',f'out_proj_{other}.')) for n in names)
    configure_branch(model,branch,2)
    out=model(torch.randn(1,4,2,2,8),torch.randn(1,5,8),torch.ones(1,5,dtype=torch.bool),frame_times=torch.arange(4)[None].float())
    out[f'updated_tokens_{branch}'].square().sum().backward()
    prefix=f'query_pool_{branch}.'
    assert any(n.startswith(prefix) and p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters())
    if branch=='event':assert model.event_temporal_reader.paths[0][1].weight.grad.norm()>0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
