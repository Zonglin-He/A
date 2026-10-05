"""Contracts for transformations whose inverse coordinate laws drive this P0."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.decota_transform_p0_v1 import *

def run():
    n=0
    ids=[10,13,17,21,24,29,32,38];native=[17,30]
    s=temporal_specs(ids,native)
    x=np.arange(8*2*3*3,dtype=np.uint8).reshape(8,2,3,3)
    for name in ['shift','crop']:
        y=temporal_pixels(x,s[name],name);assert len(y)==len(s[name]['ids']);n+=1
        off=s[name]['origin']-(s[name]['delta'] if name=='shift' else 0)
        mapped=inverse_interval([v-off for v in native],s[name],name,ids,native)
        assert mapped['interval']==native and not mapped['clipped'];n+=1
    assert np.array_equal(temporal_pixels(x,s['shift'],'shift')[s['shift']['pad']:],x);n+=1
    assert consensus([10,30],[17,39],[15,25])==[15,30];n+=1
    assert np.array_equal(spatial_pixels(spatial_pixels(x,'flip'),'flip'),x);n+=1
    b=np.array([[.2,.3,.1,.1],[.5,.5,.2,.3]])
    assert np.max(np.abs(inverse_boxes(inverse_boxes(b,'flip'),'flip')-b))<1e-15;n+=1
    assert np.allclose(per_frame_iou(b,b),1);n+=1
    invalid=inverse_interval([-100,-90],s['shift'],'shift',ids,native)
    assert invalid['fallback'] and invalid['interval']==native;n+=1
    assert not temporal_specs(ids,[10,39])['crop']['effective'];n+=1
    assert directional_query('the person on the right') and not directional_query('person runs');n+=1
    print('TRANSFORM_CONTRACTS_PASS',n)
    return n

if __name__=='__main__':run()
