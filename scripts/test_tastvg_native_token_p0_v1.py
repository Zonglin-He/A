"""Analytical non-mirroring checks: ROI support, outside and branch key layout."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_native_token_binding_v1 import roi_weights,text_query,cosine,score
def run():
 np.testing.assert_allclose(roi_weights([.25,.5,.5,1],2,2,[False]*4),[.5,0,.5,0])
 assert roi_weights([2,2,.1,.1],2,2,[False]*4) is None
 np.testing.assert_allclose(roi_weights([.5,.5,1,1],2,2,[True,False,False,False]),[0,1/3,1/3,1/3])
 H=np.zeros((10,2,256));H[4,:,0]=1;H[5,:,1]=2;m=np.zeros((2,10),bool);a=np.zeros((2,1,6));a[:,:,4]=1;a[:,:,5]=3
 q,w,mass=text_query(H,a,m,4,'spatial');np.testing.assert_allclose(q[:,0],.25);np.testing.assert_allclose(q[:,1],1.5)
 b=np.zeros((2,1,6));b[:,:,0]=3;b[:,:,1]=1;q,w,mass=text_query(H,b,m,4,'temporal');np.testing.assert_allclose(q[:,0],.75);np.testing.assert_allclose(q[:,1],.5)
 assert cosine([1,0],[0,1])==0 and cosine([0,0],[1,0]) is None
 views=[dict(H=H,info=dict(fea_map_size=[2,2],encoded_mask=m)) for _ in range(2)];z=score(views,dict(spatial=[a,a],temporal=[b,b]),np.ones((1,4,4))*.5,[0,3]);assert z['binding_scores']==[None] and z['availability'][0]['outside_frames']==0
 print('7 analytical native-token tests pass')
if __name__=='__main__':run()
