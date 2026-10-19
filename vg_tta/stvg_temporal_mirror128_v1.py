"""Training-supported reflection view; all temporal coordinates stay physical."""
import re,hashlib
import numpy as np
DIRECTION=re.compile(r'\b(?:left|right|leftward|rightward|leftmost|rightmost|east|west|clockwise|counterclockwise|anticlockwise)\b',re.I)

def eligible(caption,subject):
    # Conservatively avoid changing language meaning; no rewritten prompts.
    return DIRECTION.search(caption+' '+subject)is None

def mirror(frames):
    return frames[:,:,::-1].copy()

def fuse(p,h,m):
    assert np.array_equal(p>0,h>0)and np.array_equal(h>0,m>0)
    ens=.5*h+.5*m
    return dict(Mirror=m,MirrorEnsemble=ens,MirrorMix=.5*p+.5*ens)

def properties():
    x=np.arange(3*4*5*3,dtype=np.uint8).reshape(3,4,5,3);assert np.array_equal(mirror(mirror(x)),x)
    assert eligible('A person opens a door.','person')and not eligible('person turns left','person')and not eligible('walks','man on right')
    p=np.array([[0.,.1,.2],[0.,0.,.7],[0.,0.,0.]]);d=fuse(p,p,p);assert all(np.array_equal(v,p)for v in d.values())
    return dict(status='pass',raw_RGB_roundtrip_exact=True,direction_veto=True,identical_view_exact=True,not_an_OPD_test=True)
