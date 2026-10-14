import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import stvg_weightfree_spatial_v1 as m
import numpy as np
mask=np.zeros((64,64),bool);mask[20:40,15:30]=1
f=np.zeros((64,64,2),np.float32);f[:,:,0]=3;b=-f
out=m.warp(mask,f,b);assert np.array_equal(out[:,3:],mask[:,:-3]);assert out.sum()==mask.sum()
assert np.array_equal(m.warp(mask,np.zeros_like(f),np.zeros_like(f)),mask)
assert not m.warp(mask,f,f).any()
assert m.bbox(np.zeros_like(mask)) is None
assert m.bbox(mask)==[15/64,20/64,30/64,40/64]
assert m.iou([0,0,1,1],[0,0,1,1])==1
assert m.iou([0,0,.2,.2],[.3,.3,.4,.4])==0
tracks=[dict(id=1,boxes=[[.1,.1,.5,.5],None])];frames=[np.zeros((64,64,3),np.uint8)]*2
v=m.render(frames,tracks,[],{'1':1});assert v.shape==(2,3,448,448) and v.dtype==np.uint8
m.write(m.BASE/'CPU_CONTRACTS.json',dict(status='pass',count=8,GT_read=False,CPU_only=True,scope='translation/zero-flow/cycle-reject/empty/box/IoU/render'))
print('8 synthetic semantic contracts passed')
