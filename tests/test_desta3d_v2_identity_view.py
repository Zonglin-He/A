import copy
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_identity_view_tta import adapt_identity_view
from vg_tta.desta3d_v2_tta_pilot import adapt


def test_identity_keeps_old_recipe_and_regularizes_after_step_one():
    torch.set_num_threads(2); torch.manual_seed(91)
    a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).eval()
    b=copy.deepcopy(a); initial=copy.deepcopy(a.state_dict())
    f={'visual_grid':torch.randn(1,3,2,2,2560),'query_tokens':torch.randn(1,4,2560),
       'query_mask':torch.ones(1,4,dtype=torch.bool),'frame_times':torch.tensor([[0.,.5,1.]])}
    moments={'feature_definition':'v2_query_conditioned_readers',
             **{k:{'mean':[0.]*128,'std':[1.]*128} for k in ['event','spatial']}}
    result=adapt_identity_view(a,f,moments=moments)
    reference=adapt(b,f,copy.deepcopy(f),interface='calibration',alignment=.01,moments=moments,steps=3,lr=1e-5)
    assert result['history']==reference['history']
    assert all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items())
    for term in ['latent','referent','event','parameter_anchor']:
        assert abs(result['history'][0]['terms_before'][term])<1e-7
    assert result['history'][0]['terms_before']['alignment']>0
    assert result['history'][2]['terms_before']['latent']>0
    assert result['history'][2]['terms_before']['parameter_anchor']>0
    assert [h['actual_Adam_steps'] for h in result['history']]==[[1],[2],[3]]
    assert result['parameter_count']==66816 and not result['output_anchor']
    assert all(not p.requires_grad and p.grad is None for p in a.parameters())
    a.load_state_dict(initial)
    assert all(torch.equal(v,initial[n]) for n,v in a.state_dict().items())
