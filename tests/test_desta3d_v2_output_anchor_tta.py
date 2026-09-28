import copy
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_tta_pilot import adapt
from vg_tta.desta3d_v2_output_anchor_tta import adapt_output_anchor,teacher_support


def test_disabled_output_factor_exactly_matches_existing_three_steps():
    torch.set_num_threads(2);torch.manual_seed(17)
    a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False)
    b=copy.deepcopy(a)
    fields={'visual_grid':torch.randn(1,3,2,2,2560),'query_tokens':torch.randn(1,4,2560),
            'query_mask':torch.ones(1,4,dtype=torch.bool),'frame_times':torch.tensor([[0.,.5,1.]])}
    view=dict(fields,visual_grid=fields['visual_grid']*.91+.02)
    moments={'feature_definition':'v2_query_conditioned_readers',
             **{k:{'mean':[0.]*128,'std':[1.]*128} for k in ['event','spatial']}}
    adapt(a,fields,view,interface='calibration',alignment=.01,moments=moments)
    rows=[]
    def forbidden(*a,**k):raise AssertionError('missing teacher must not fabricate output')
    report=adapt_output_anchor(torch.nn.Linear(2,2).requires_grad_(False),None,b,fields,view,{'branches':[]},[],
            moments=moments,coefficient=6.2,save_step=lambda s,x:rows.append(x),replay=forbidden)
    assert report['steps']==3 and report['gates_unchanged'] and len(rows)==3
    assert all(torch.equal(x,b.state_dict()[n]) for n,x in a.state_dict().items())


def test_teacher_failure_and_invalid_geometry_policy():
    trace={'branches':[{'logits':{'time':torch.zeros(2,32)}},{'logits':{'coordinate':torch.zeros(2,4,1001)}}]}
    assert teacher_support({'event':{'format_ok':False}},trace)==[]
    assert teacher_support({'event':{'format_ok':True},'spatial':{'geometry_valid':[False,False]}},trace)==['event','spatial']
    trace['branches'][1]['logits']['coordinate'][0,0,0]=float('nan')
    assert teacher_support({'event':{'format_ok':True}},trace)==['event']
