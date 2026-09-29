import copy
import torch
from vg_tta.tastvg_paper_online_baselines_v1 import OnlineOptimizer
from tests.test_vitta_paper_v1 import toy,source


def test_tent_persists_and_resume_matches_next_arrival():
 p,ids,views=toy();base=p.detach().clone();closure=lambda:[v.native for v in views()]
 policy=OnlineOptimizer([p],'TENT');pre,post,a=policy.arrive(closure,lambda:p.detach().clone())
 assert a['parameter_delta']>0 and not torch.equal(post,base)
 snapshot=policy.state_dict();pre2,post2,_=policy.arrive(closure,lambda:p.detach().clone())
 assert torch.equal(pre2,post)
 policy.load_state_dict(snapshot);_,replay,_=policy.arrive(closure,lambda:p.detach().clone())
 assert torch.equal(replay,post2)
 policy.reset();assert torch.equal(p,base) and len(policy.optimizer.state)==0


def test_vitta_ema_survives_arrivals_and_detaches_history():
 p,ids,views=toy();policy=OnlineOptimizer([p],'ViTTA',source=source())
 policy.arrive(views,lambda:p.detach().clone(),ids);old=copy.deepcopy(policy.target)
 current=__import__('vg_tta.vitta_paper_v1',fromlist=['feature_statistics']).feature_statistics(views())
 policy.arrive(views,lambda:p.detach().clone(),ids)
 for k in old:
  for q in old[k]:
   torch.testing.assert_close(policy.target[k][q],.9*old[k][q]+.1*current[k][q])
   assert policy.target[k][q].grad_fn is None
 assert policy.arrivals==2
 policy.reset();assert policy.target is None


def test_sar_high_entropy_skip_is_reported_without_fake_update():
 p,ids,views=toy();policy=OnlineOptimizer([p],'SAR',config={'margin_fraction':0.})
 pre,post,a=policy.arrive(lambda:[v.native for v in views()],lambda:p.detach().clone())
 assert torch.equal(pre,post) and all(not x['updated'] for x in a['trace'])
