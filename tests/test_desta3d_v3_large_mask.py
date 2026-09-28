import copy
import pytest
import torch
from tests.test_desta3d_v3_latent_oracle import fixture, label
from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record
from vg_tta.desta3d_v3_location_probe import mask_at_location
from vg_tta.desta3d_v3_free_actuation import inject_delta
from vg_tta.desta3d_v3_large_mask import large_mask_delta, RATIOS

@pytest.mark.parametrize('branch', ['event', 'spatial'])
def test_real_hidden128_fixed_magnitude_direction_scope_and_restore(branch):
    a, f, x = fixture(); state=copy.deepcopy(a.state_dict()); base=f()
    masks=masks_from_source_record(label(), label()['frame_ids'], 2, 2)
    with mask_at_location(a, branch, masks[branch], 'late'):
        candidate=f()
    b=base['updated_tokens_'+branch]
    raw, delta, log=large_mask_delta(x,b,candidate['updated_tokens_'+branch],branch)
    assert log['applicable']
    assert abs(float(delta.double().norm())/float(x.double().norm())-RATIOS[branch])<1e-7
    assert torch.equal(delta,raw*log['scale']) and log['scale']>0
    with inject_delta(a,branch,delta):
        out=f()
    assert torch.equal(out['updated_tokens_'+branch],b+delta)
    other='spatial' if branch=='event' else 'event'
    assert torch.equal(out['updated_tokens_'+other],base['updated_tokens_'+other])
    with pytest.raises(RuntimeError):
        with inject_delta(a,branch,delta):raise RuntimeError('synthetic')
    assert torch.equal(f()['updated_tokens_'+branch],b)
    assert all(torch.equal(v,state[k]) for k,v in a.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in a.parameters())

def test_denominator_is_stock_and_correct_wrong_are_independently_matched():
    stock=torch.arange(1,25,dtype=torch.float32).reshape(1,2,2,2,3)
    b=stock+8.;good=b+torch.linspace(-.02,.04,24).reshape_as(b);wrong=b+.03
    _,d1,l1=large_mask_delta(stock,b,good,'event')
    _,d2,l2=large_mask_delta(stock,b,wrong,'event')
    assert l1['target_norm']==l2['target_norm']==RATIOS['event']*float(stock.double().norm())
    assert l1['target_norm']!=RATIOS['event']*float(b.double().norm())
    assert abs(float(d1.double().norm()-d2.double().norm()))<1e-6
    assert l1['scale']!=l2['scale']

def test_neutral_and_invalid_support():
    x=torch.ones(1,2,2,2,4)
    raw,d,l=large_mask_delta(x,x,x,'event')
    assert not l['applicable'] and l['requested_norm']>0 and l['target_norm']==0 and not d.any()
    with pytest.raises(ValueError):large_mask_delta(x,x,x[:,:1],'event')
    with pytest.raises(ValueError):large_mask_delta(x,x,x*float('nan'),'event')
    with pytest.raises(ValueError):large_mask_delta(x*0,x,x,'event')
    with pytest.raises(ValueError):large_mask_delta(x,x,x,'unknown')

def test_full_FP32_addition_and_BF16_cast_can_be_reconstructed():
    stock=torch.linspace(-2,2,48).reshape(1,2,2,2,6)
    base=stock+.001;candidate=base+torch.sin(stock)*.004
    _,d,_=large_mask_delta(stock,base,candidate,'spatial')
    before=base.bfloat16();after=(base+d).bfloat16();changed=before!=after
    restored=before.clone();restored[changed]=after[changed]
    assert torch.equal(restored,after) and changed.any()
