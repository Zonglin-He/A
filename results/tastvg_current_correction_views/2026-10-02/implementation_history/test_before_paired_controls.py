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
    print('8 analytic boundary controls passed')
if __name__=='__main__':run()
