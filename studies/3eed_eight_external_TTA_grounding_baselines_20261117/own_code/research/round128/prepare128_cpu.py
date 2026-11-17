"""Create baseline-only physical region data; never read target annotations."""
from pathlib import Path
import json,hashlib,sys,numpy as np
from scipy.spatial import cKDTree
from PIL import Image
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128';rd=lambda p:json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def example(z,g,y=None):
 xyz=z['xyz'][0];boxes=z['boxes'];crops=[];images=[];vis=[];im=Image.fromarray(g['image']);uv=g['uv_grid']*8
 # Crops use actual seed points; unknown/empty supports are explicitly recorded.
 for i,b in enumerate(boxes):
  ins=np.flatnonzero((abs(xyz-b[:3])<=np.maximum(b[3:6],.05)/2).all(1));valid=len(ins)>0
  if not valid:ins=np.argsort(((xyz-b[:3])**2).sum(1))[:16]
  chosen=ins[np.arange(128)%len(ins)] if len(ins)<128 else ins[np.linspace(0,len(ins)-1,128,dtype=int)]
  crops.append(np.clip((xyz[chosen]-b[:3])/max(float(np.linalg.norm(b[3:6])/2),.1),-4,4))
  vi=ins[g['visible'][ins]] if valid else np.array([],int);ok=len(vi)>0
  if ok:
   lo=np.maximum(np.floor(uv[vi].min(0))-4,0).astype(int);hi=np.minimum(np.ceil(uv[vi].max(0))+5,[384,256]).astype(int)
   ok=bool((hi>lo).all())
  images.append(np.asarray(im.crop((*lo,*hi)).resize((24,24),Image.Resampling.BILINEAR)) if ok else np.zeros((24,24,3),np.uint8));vis.append(ok)
 q=z['query'].astype(np.float32);u=z['tokens'];mask=z['token_mask'].astype(bool);text=u[mask].mean(0);text/=max(float(np.linalg.norm(text)),1e-8);joint=q*text;native=q@u.T/.07;s=np.logaddexp.reduce(native[:,mask],1)-np.logaddexp.reduce(native[:,~mask],1)
 out={'crops':np.asarray(crops,np.float32),'images':np.asarray(images,np.uint8),'visible':np.array(vis),'joint':joint,'text':text.astype(np.float32),'native_scores':s,'boxes':boxes,'h':z['h'].astype(np.float32)}
 if y is not None:out['source_IoU']=y
 return out
def main():
 registry=[]
 for src in ['waymo','drone','quad']:
  for split,n in [('TRAIN512',192),('VAL128',64)]:
   rows=rd(R/(f'research/round76/input/SOURCE_{split}_PRIVATE.json' if src=='waymo' else f'research/round97/input/SOURCE_{src.upper()}_{split}_PRIVATE.json'))[:n]
   for i,r in enumerate(rows):
    sid=r['id']
    if src=='waymo':zp=R/f'crossplatform/round76/source_prefix640_attempt001/samples_PRIVATE/{sid}.npz';gp=R/f'crossplatform/round76/prepared976_attempt001/samples_PRIVATE/{sid}.npz';yp=R/f'research/round25/experiment/source_labels_attempt002/samples_PRIVATE/{sid}.npz'
    else:root=R/f'crossplatform/round97/{src}/samples_PRIVATE';zp=root/(sid+'_prefix.npz');gp=root/(sid+'_geometry.npz');yp=root/(sid+'_source_labels.npz')
    dest=B/f'regions_PRIVATE/source/{src}/{split}/{sid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():np.savez_compressed(dest,**example(load(zp),load(gp),load(yp)['ious']))
    registry.append({'role':'source','source':src,'split':split,'path':str(dest.relative_to(R)),'sha256':sha(dest)})
    if i%32==0:print('R128_REGIONS_SOURCE',src,split,i,flush=True)
 for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json'):
  if e['cohort']!='DEV' or e['protected_alias_fallback']:continue
  z=load(R/e['prefix']);key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();gp=R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(e['row']['id']+'_'+key[:20]+'.npz');dest=B/f"regions_PRIVATE/target/{e['source']}_to_{e['target']}/{e['row']['id']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():np.savez_compressed(dest,**example(z,load(gp)))
  registry.append({'role':'target','source':e['source'],'target':e['target'],'id_PRIVATE':e['row']['id'],'path':str(dest.relative_to(R)),'sha256':sha(dest)})
 (H/'REGION_REGISTRY_PRIVATE.json').write_text(json.dumps(registry,indent=2)+'\n');print('R128_REGIONS_COMPLETE',len(registry),flush=True)
if __name__=='__main__':main()
