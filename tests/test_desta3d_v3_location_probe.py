import copy,pytest,torch
from tests.test_desta3d_v3_latent_oracle import fixture,label
from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record
from vg_tta.desta3d_v3_location_probe import mask_at_location,match_delta

@pytest.mark.parametrize('branch',['event','spatial'])
def test_real_hidden128_identity_scope_location_and_exception_restore(branch):
 a,f,x=fixture();state=copy.deepcopy(a.state_dict());base=f();ones=torch.ones(x.shape[:-1]);m=masks_from_source_record(label(),label()['frame_ids'],2,2)[branch]
 for location in ['early','late']:
  with mask_at_location(a,branch,ones,location):same=f()
  assert torch.equal(same['updated_tokens_'+branch],base['updated_tokens_'+branch])
  with mask_at_location(a,branch,m,location):changed=f()
  other='event' if branch=='spatial' else 'spatial'
  assert torch.equal(changed['updated_tokens_'+other],base['updated_tokens_'+other])
  assert not torch.equal(changed['updated_tokens_'+branch],base['updated_tokens_'+branch])
  with pytest.raises(RuntimeError):
   with mask_at_location(a,branch,m,location):raise RuntimeError('synthetic')
 assert all(torch.equal(v,state[k]) for k,v in a.state_dict().items())
 assert all(p.grad is None and not p.requires_grad for p in a.parameters())
 assert torch.equal(f()['updated_tokens_'+branch],base['updated_tokens_'+branch])

def test_matching_sign_zero_and_nonfinite_contract():
 b=torch.tensor([1.,2.,3.]);c=b+torch.tensor([.002,-.003,.001]);raw,d,log=match_delta(b,c,.004)
 assert torch.dot(raw,d)>0 and abs(float(d.double().norm())-.004)<1e-8
 assert abs(log['realized_norm']-.004)<4e-6
 assert torch.equal(match_delta(b,c,0)[1],torch.zeros_like(b))
 with pytest.raises(ValueError,match='zero intervention'):match_delta(b,b,1.)
 with pytest.raises(ValueError,match='Nonfinite'):match_delta(b,c*float('nan'),1.)

def test_early_really_precedes_LN_and_late_follows_it():
 a,f,x=fixture();m=masks_from_source_record(label(),label()['frame_ids'],2,2)['event'];seen={}
 for loc in ['early','late']:
  h=a.norm_event.register_forward_pre_hook(lambda mod,args:seen.update({loc:args[0].clone()}))
  with mask_at_location(a,'event',m,loc):f()
  h.remove()
 assert not torch.equal(seen['early'],seen['late'])
 # Positive common-channel scaling is approximately erased by channel-only LN.
 z=torch.randn(1,128,3,2,2);assert torch.allclose(a.norm_event(z),a.norm_event(z*.5),atol=2e-4,rtol=2e-4)

@pytest.mark.parametrize('branch',['event','spatial'])
def test_matched_injection_exact_late_replay(branch):
 from vg_tta.desta3d_v3_free_actuation import inject_delta
 a,f,x=fixture();base=f();m=masks_from_source_record(label(),label()['frame_ids'],2,2)[branch]
 with mask_at_location(a,branch,m,'late'):late=f()
 raw=late['updated_tokens_'+branch]-base['updated_tokens_'+branch];target=float(raw.double().norm())
 for loc in ['late','early']:
  with mask_at_location(a,branch,m,loc):candidate=f()
  _,delta,log=match_delta(base['updated_tokens_'+branch],candidate['updated_tokens_'+branch],target)
  with inject_delta(a,branch,delta):matched=f()
  if loc=='late':assert torch.equal(matched['updated_tokens_'+branch],late['updated_tokens_'+branch])
  other='event' if branch=='spatial' else 'spatial'
  assert torch.equal(matched['updated_tokens_'+other],base['updated_tokens_'+other])
