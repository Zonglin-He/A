"""Analytic controls for candidate ties, empty evidence and view disagreement."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vg_tta.tastvg_current_correction_views_v1 import select,temporal,uniform_second
def run():
    assert select(None)['selected']==0
    assert select([0,1,1])['selected']==0
    assert select([0,1,2])['selected']==2
    assert select([0,0,0])['selected']==0
    c=[{'physical_interval':[0,10]},{'physical_interval':[2,8]}]
    a={'proposals':[[2,8]],'proposal_confidence':[1.]};b={'proposals':[[0,10]],'proposal_confidence':[1.]}
    assert temporal(c,[a,a])['selected']==1
    assert temporal(c,[a,b])['selected']==0
    assert temporal(c,[dict(proposals=[],proposal_confidence=[]),a])['selected']==0
    assert len(set(uniform_second(22)))==5
    from scripts.diagnose_tastvg_correction_views_v1 import source_values,effect
    rows=[dict(source_id=s,order=o,condition=c,L_v=d,R_v=0.) for s,o,c,d in
          [(0,'one','a',1.),(0,'one','b',3.),(0,'two','a',5.),(1,'one','a',-1.)]]
    ids,v=source_values(rows,'L','R');assert ids==[0,1] and v.tolist()==[3.5,-1.]
    z=effect(rows,'L','L');assert z['mean']==0. and z['ci95']==[0.,0.] and z['leave_one_source_out_range']==[0.,0.]
    print('10 analytic controls passed: 8 readout boundaries and 2 paired aggregation controls')
if __name__=='__main__':run()
