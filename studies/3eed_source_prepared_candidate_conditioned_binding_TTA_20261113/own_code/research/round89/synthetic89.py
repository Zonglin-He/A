from core89 import *
torch.manual_seed(89);m=CandidateInterface().eval();E=3;h=torch.randn(256,288);local=torch.randn(E,3,288);own=torch.randn(256,288);text=torch.randn(E,288);rel=torch.randn(E,288);route={'weights_plain':torch.ones(E,3)/3,'hyp_geo':torch.randn(256,E,3,9),'own_geo':torch.randn(256,9),'own_valid':torch.ones(256,dtype=torch.bool),'relation_tokens':torch.ones(E,8,dtype=torch.bool)}
with torch.no_grad():
 assert torch.equal(m(h,local,own,text,rel,route),torch.zeros(256));m.out[-1].weight.normal_(std=.1);m.out[-1].bias.fill_(.1);a=m(h,local,own,text,rel,route);p=torch.randperm(256);rt={**route,'hyp_geo':route['hyp_geo'][p],'own_geo':route['own_geo'][p],'own_valid':route['own_valid'][p]};b=m(h[p],local,own[p],text,rel,rt);queryerr=float(abs(a[p]-b).max());assert queryerr<1e-5
 p=torch.tensor([2,0,1]);rt={**route,'weights_plain':route['weights_plain'][:,p],'hyp_geo':route['hyp_geo'][:,:,p]};b=m(h,local[:,p],own,text,rel,rt);hyperr=float(abs(a-b).max());assert hyperr<1e-5
 c=m(h,local.flip(0),own,text,rel,route);assert float(abs(a-c).max())>1e-5
assert not torch.cuda.is_initialized();dump(H89/'SYNTHETIC_AUDIT.json',{'state':'PASS_R89_ZERO_RESIDUAL_QUERY_EQUIVARIANCE_HYPOTHESIS_EQUIVARIANCE_ROLE_SENSITIVITY','query_permutation_error':queryerr,'hypothesis_permutation_error':hyperr,'task_GT_read':False});print('PASS')
