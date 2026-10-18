import numpy as np,torch
from vg_tta.stvg_interval_mbr_v1 import expected_iou,legal_distribution,mbr
from vg_tta.stvg_identity_trajectory_opd_v1 import covariance,log_density,reward

def test_mbr_is_expected_metric_not_map():
 iv=np.array([[4,7],[2,9],[1,10]]);q=np.array([.4,.3,.3]);u=expected_iou(iv,q)
 brute=np.array([sum(p*max(0,min(b,d)-max(a,c))/((b-a)+(d-c)-max(0,min(b,d)-max(a,c)))for(c,d),p in zip(iv,q))for a,b in iv])
 np.testing.assert_allclose(u,brute,atol=1e-15);assert int(u.argmax())!=int(q.argmax())

def test_mbr_all_legal_in_physical_time():
 z=np.array([[1,0],[0,2],[3,-1],[0,4]]);ids=[0,2,9,14];iv,q,ij=legal_distribution(z,ids)
 assert len(iv)==6 and (ij[:,0]<ij[:,1]).all();out=mbr(z,ids);assert out['expected_tIoU']>=out['map_expected_tIoU']-1e-14

def test_joint_gaussian_density_and_gradient():
 torch.manual_seed(4);n=5;mean=torch.randn(n,4,dtype=torch.float64,requires_grad=True);K=covariance(list(range(n)),mean.device,mean.dtype,True);L=torch.linalg.cholesky(K);inv=torch.cholesky_inverse(L);ld=2*L.diag().log().sum();sigma=.2
 offsets=torch.randn(3,n,4,dtype=mean.dtype).detach();lw=torch.log_softmax(torch.randn(3,dtype=mean.dtype),0);samples=torch.randn(7,n,4,dtype=mean.dtype)
 actual=log_density(samples,mean,offsets,lw,inv,ld,sigma)
 terms=[]
 for h in range(3):
  vals=sum(torch.distributions.MultivariateNormal(mean[:,c]+offsets[h,:,c],covariance_matrix=sigma**2*K).log_prob(samples[:,:,c])for c in range(4));terms.append(lw[h]+vals)
 expect=torch.logsumexp(torch.stack(terms,1),1);torch.testing.assert_close(actual,expect,atol=1e-9,rtol=1e-12)
 g=torch.autograd.grad(actual.sum(),mean)[0];eps=1e-6;m=mean.detach().clone();m[1,2]+=eps;p=log_density(samples,m,offsets,lw,inv,ld,sigma).sum();m[1,2]-=2*eps;minus=log_density(samples,m,offsets,lw,inv,ld,sigma).sum();assert abs(float((p-minus)/(2*eps)-g[1,2]))<1e-5
 torch.testing.assert_close(K.diag(),torch.ones(n,dtype=mean.dtype));assert float(torch.linalg.eigvalsh(K).min())>0

def test_no_per_frame_identity_splicing():
 # Two coherent people. A switched identity may not get the per-frame oracle1.
 a=torch.tensor([[.2,.5,.15,.3],[.2,.5,.15,.3]]);b=a.clone();b[:,0]=.8
 h=dict(positions=[0,1],targets=torch.stack([a,b]),mask=torch.ones(2,2,dtype=torch.bool),temporal_weights=torch.tensor([.5,.5]),logw=torch.tensor([.5,.5]).log())
 mixed=torch.stack([a[0],b[1]]);assert float(reward(mixed,h,.1))<float(reward(a,h,.1));assert float(reward(mixed,h,.1))<.6
