import copy
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_signal_probe import measure_signals


def test_same_observation_zero_consistency_and_no_update():
    torch.set_num_threads(2);torch.manual_seed(91)
    a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).eval()
    state=copy.deepcopy(a.state_dict())
    f={'visual_grid':torch.randn(1,3,2,2,2560),'query_tokens':torch.randn(1,4,2560),
       'query_mask':torch.ones(1,4,dtype=torch.bool),'frame_times':torch.tensor([[0.,.5,1.]])}
    moments={'feature_definition':'v2_query_conditioned_readers',
             **{k:{'mean':[0.]*128,'std':[1.]*128} for k in ['event','spatial']}}
    native=measure_signals(a,f,f,moments)
    for n in ['latent','referent','event','parameter_anchor']:
        assert abs(native['terms'][n])<1e-7
        assert native['gradient_norms'][n]<1e-6
    assert native['gradient_norms']['alignment']>0
    mild=measure_signals(a,f,dict(f,visual_grid=f['visual_grid']*.91+.02),moments)
    assert mild['gradient_norms']['latent']>0
    assert all(torch.equal(v,a.state_dict()[n]) for n,v in state.items())
    assert not any(p.grad is not None or p.requires_grad for p in a.parameters())
