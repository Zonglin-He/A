"""CPU controls for the changed branch selection, not real PTD efficacy."""
import copy
import pytest
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_output_anchor_tta import adapt_output_anchor
from vg_tta.desta3d_v2_temporal_anchor_tta import adapt_temporal_anchor, temporal_teacher_support


def test_temporal_support_retains_event_when_coordinate_missing():
    trace = {'branches': [{'logits': {'time': torch.zeros(2, 3)}}, {'logits': {}}]}
    assert temporal_teacher_support({'event': {'format_ok': True}}, trace) == ['event']
    assert temporal_teacher_support({'event': {'format_ok': False}}, trace) == []
    trace['branches'][1]['logits']['coordinate'] = torch.zeros(2, 4, 1001)
    assert temporal_teacher_support({'event': {'format_ok': True}}, trace) == ['event']


def test_event_only_matches_original_enabled_contract_and_rejects_spatial():
    torch.set_num_threads(2); torch.manual_seed(28)
    a = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False)
    b = copy.deepcopy(a)
    fields = {'visual_grid': torch.randn(1,3,2,2,2560), 'query_tokens': torch.randn(1,4,2560),
              'query_mask': torch.ones(1,4,dtype=torch.bool), 'frame_times': torch.tensor([[0.,.5,1.]])}
    view = dict(fields, visual_grid=fields['visual_grid']*.91+.02)
    moments = {'feature_definition':'v2_query_conditioned_readers',
               **{k:{'mean':[0.]*128,'std':[1.]*128} for k in ['event','spatial']}}
    trace = {'branches': [{'logits': {'time': torch.zeros(2,3)}}]}
    calls = []
    def synthetic_replay(model, prompt, adapter, fields, trace, branch):
        assert branch == 'event'; calls.append(branch)
        z = adapter(fields['visual_grid'],fields['query_tokens'],frame_times=fields['frame_times'],query_mask=fields['query_mask'])
        scalar = z['branch_features_event'].square().mean()
        return torch.stack([scalar, -scalar, scalar*0]).repeat(2,1), {}
    model = torch.nn.Linear(2,2).requires_grad_(False)
    records = [[], []]
    for i, (fn, adapter) in enumerate([(adapt_output_anchor,a),(adapt_temporal_anchor,b)]):
        fn(model,None,adapter,fields,view,trace,['event'],moments=moments,coefficient=6.248522551708088,
           save_step=lambda s,x,i=i:records[i].append(x),replay=synthetic_replay)
    assert len(calls) == 6
    assert all(torch.equal(v, b.state_dict()[n]) for n,v in a.state_dict().items())
    assert all('spatial_weighted' not in r['raw_gradients'] for arm in records for r in arm)
    assert all(r['history']['actual_Adam_steps'] == [i] for arm in records for i,r in enumerate(arm,1))
    with pytest.raises(ValueError, match='spatial output'):
        adapt_temporal_anchor(model,None,b,fields,view,trace,['event','spatial'],moments=moments,
                              coefficient=6.248522551708088,save_step=lambda *x:None)
