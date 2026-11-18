from pathlib import Path
import sys,numpy as np
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(P/'upstream')]
from geometry_refine import reconstruct_points

def refine(points,src):
 source7=np.r_[src[:6],np.float32(0)].astype(np.float32);refined=source7.copy();points=np.unique(points,axis=0);selected=np.all(abs(points-src[:3])<=src[3:6]/2+1e-7,axis=1);crop=points[selected];reason='updated';values=None
 if len(crop)<8:reason='fewer_than8_crop_points'
 elif src[3]/src[4]<1.5:reason='predicted_L_over_W_below1.5'
 else:
  xy=crop[:,:2].astype(float);cov=np.cov(xy,rowvar=False);values,vectors=np.linalg.eigh(cov)
  if values[-1]<=1e-12:reason='degenerate_XY_covariance'
  else:
   direction=vectors[:,-1];yaw=np.arctan2(direction[1],direction[0]);refined[6]=((yaw+np.pi/2)%np.pi)-np.pi/2
 assert np.array_equal(source7[:6],refined[:6]);return source7,refined,{'unique_visible_points':len(points),'predicted_crop_points':len(crop),'predicted_L_over_W':float(src[3]/src[4]),'eigenvalues_XY':values.tolist() if values is not None else None,'yaw_radians':float(refined[6]),'reason':reason,'center_and_size_bitexact':True}
