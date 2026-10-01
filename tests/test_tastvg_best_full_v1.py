import ast,tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.tastvg_best_full_common_v1 import savez,loadz
from scripts.score_tastvg_best_full_v1 import source_summary,geom
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry

def test_logging_does_not_change_single_step_math():
 a=ast.parse(Path('vg_tta/tastvg_extended_method_v3.py').read_text());b=ast.parse(Path('vg_tta/tastvg_best_full_method_v1.py').read_text())
 for name in ['SingleStep','reverse_kl','rollout_states','nested_directions']:
  x=next(n for n in a.body if getattr(n,'name',None)==name);y=next(n for n in b.body if getattr(n,'name',None)==name);assert ast.dump(x)==ast.dump(y)
def test_lossless_compressed_tensors():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'x.pt.gz';x={'states':torch.randn(1792),'candidate':torch.rand(9,64,4)};savez(p,x);y=loadz(p);assert all(torch.equal(x[k],y[k]) for k in x)
def test_source_weighting_is_not_query_weighting():
 rows=[dict(source_id=s,order=o,condition=c,delta_v=v) for s,vals in [(0,[0.,0.,0.]),(1,[1.])] for v in vals for o in ['a','b'] for c in ['x','y']]
 z=source_summary(rows,['delta_v']);assert z['sources']==2 and z['metrics']['delta_v']['mean']==.5 and z['metrics']['delta_v']['query_macro']==.25
 assert z['metrics']['delta_v']['order_values']==[.5,.5]
def test_independent_geometry():
 g=torch.Generator().manual_seed(1);p=torch.rand(64,4,generator=g)*.5+.2;q=torch.rand(9,64,4,generator=g)*.5+.2
 np.testing.assert_allclose(geom(p,q,[5,2]),geometry(p,q,5,2).numpy(),atol=2e-6,rtol=2e-6)
