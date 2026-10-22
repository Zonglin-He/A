import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from tastvg_paper_score import cluster

def test_parent_cluster_point_estimate_stays_equal_query():
    rows=[dict(parent=p,values=dict(A={'v':x},B={'v':0.}))for p,x in [('one',.1),('one',.1),('one',.1),('two',.5)]]
    z=cluster(rows,'A','B',[('v',100)])['v']
    assert abs(z['mean']-20.)<1e-12 and z['parents']==2 and z['queries']==4

def test_lower_error_and_missing_are_improvements():
    rows=[dict(parent='p',values=dict(A={'missing':0.,'boundary_MAE_seconds':1.},B={'missing':1.,'boundary_MAE_seconds':2.}))]
    z=cluster(rows,'A','B',[('missing',100),('boundary_MAE_seconds',1)])
    assert all(x['improved']==1 and x['harmed']==0 for x in z.values())
