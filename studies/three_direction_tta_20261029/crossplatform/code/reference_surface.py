"""Fixed three-yaw surface-energy selection. No GT/image/network update."""
import numpy as np

def rotation(yaw):
 c,s=np.cos(yaw),np.sin(yaw);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def surface_loss(points,box):
 local=(points.astype(float)-box[:3])@rotation(float(box[6]));q=abs(local)-box[3:6]/2
 sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
 return float(np.abs(sdf).mean()/min(box[3],box[4]))
def refine(points,source7,pca7):
 source7=np.asarray(source7,np.float32);pca7=np.asarray(pca7,np.float32);assert np.array_equal(source7[:6],pca7[:6]);points=np.unique(points,axis=0);crop=points[np.all(abs(points-source7[:3])<=source7[3:6]/2+1e-7,axis=1)];reason='selected_by_surface_loss';candidates=[source7.copy(),pca7.copy(),pca7.copy()];candidates[2][6]=((float(pca7[6])+np.pi/2+np.pi/2)%np.pi)-np.pi/2
 if len(crop)<8:reason='fewer_than8_crop_points'
 elif source7[3]/source7[4]<1.5:reason='predicted_L_over_W_below1.5'
 else:
  values=np.linalg.eigvalsh(np.cov(crop[:,:2].astype(float),rowvar=False))
  if values[-1]<=1e-12:reason='degenerate_XY_covariance'
 if reason!='selected_by_surface_loss':return source7.copy(),{'reason':reason,'selected_candidate':0,'crop_points':len(crop),'candidate_losses':None}
 losses=[surface_loss(crop,b.astype(float)) for b in candidates];minimum=min(losses);idx=next(i for i,x in enumerate(losses) if x<=minimum+1e-12);result=candidates[idx].copy();assert np.array_equal(result[:6],source7[:6]);return result,{'reason':reason,'selected_candidate':idx,'crop_points':len(crop),'candidate_losses':losses}
