"""Global surface graphs precede candidate association; components are not instances."""
from common72 import *
import sys
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from attributes51 import rgb_to_lab
sys.path.insert(0,str(ROOT/'research/round53'));from support53 import pixel_descriptors

def scene(r):
 raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy();xyz=raw.copy()
 if r['platform']=='drone':xyz[:,2]+=np.float32(1.8)
 rgb=np.array(Image.open(r['imagepath']).convert('RGB'));HH,W=rgb.shape[:2];cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');pr=project_model_points(xyz.astype(float),cam,(W,HH));ii=np.flatnonzero(pr['in_image']);xy=np.floor(pr['uv'][ii]).astype(int);cell=(xy[:,1]//4)*((W+3)//4)+xy[:,0]//4;order=np.lexsort((pr['depth'][ii],cell));_,first=np.unique(cell[order],return_index=True);ids=ii[order[first]]
 if len(ids)>4096:ids=ids[np.linspace(0,len(ids)-1,4096,dtype=int)]
 x=xyz[ids].astype(float);uv=pr['uv'][ids];depth=pr['depth'][ids];pix=np.floor(uv).astype(int);color=rgb[pix[:,1],pix[:,0]];n=len(ids)
 if n<12:return {'xyz':x,'uv':uv,'depth':depth,'rgb':color,'labels':np.tile(np.arange(n),(2,1)),'edge_ij':np.empty((0,2),int),'edge_masks':np.empty((2,0),bool),'normal':np.zeros((n,3)),'raw_indices':ids,'wh':np.array([W,HH]),'lab':rgb_to_lab(color)}
 dist,kn=cKDTree(x).query(x,k=12,workers=1);local=x[kn]-x[kn].mean(1,keepdims=True);eig,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',local,local)/12);norm=vec[:,:,0];stable=(eig[:,1]-eig[:,0])/np.maximum(eig[:,2],1e-9)>.15;src=np.repeat(np.arange(n),11);dst=kn[:,1:].reshape(-1);d=dist[:,1:].reshape(-1);spacing=np.maximum(dist[:,1],.02);dlimit=np.clip(3*np.maximum(spacing[src],spacing[dst]),.06,.8);sep=np.linalg.norm(uv[src]-uv[dst],axis=1);depthok=np.abs(depth[src]-depth[dst])<.15+.025*np.minimum(depth[src],depth[dst]);normaldot=np.abs((norm[src]*norm[dst]).sum(1));lab=rgb_to_lab(color);line=np.stack([uv[src]*(1-t)+uv[dst]*t for t in [0.,.25,.5,.75,1.]],1);p=np.floor(line).astype(int);line_lab=rgb_to_lab(rgb[p[:,:,1],p[:,:,0]].reshape(-1,3)).reshape(len(src),5,3);jump=np.linalg.norm(np.diff(line_lab,axis=1),axis=2).max(1);edges=[];labels=[]
 for labmax,cos in [(20.,np.cos(np.pi/6)),(30.,np.cos(np.pi/4))]:
  ok=(d<=dlimit)&(sep<=32)&depthok&stable[src]&stable[dst]&(normaldot>=cos)&(jump<=labmax);mat=coo_matrix((np.ones(int(ok.sum())),(src[ok],dst[ok])),shape=(n,n)).tocsr();_,cc=connected_components(mat,directed=False);labels.append(cc);edges.append(ok)
 return {'xyz':x,'uv':uv,'depth':depth,'rgb':color,'labels':np.array(labels),'edge_ij':np.c_[src,dst],'edge_masks':np.array(edges),'normal':norm,'raw_indices':ids,'wh':np.array([W,HH]),'lab':lab}

def supports(s,boxes,a,request):
 nG=len(a['rects']);n=len(s['xyz']);known=np.zeros(nG,bool);desc=np.zeros((nG,13));high=np.zeros((nG,13));all_rgb=np.zeros((nG,13));box_known=np.zeros(nG,bool);counts=np.zeros(nG,int);components=np.zeros((2,nG),int);coverage=np.zeros((2,nG));xs=s['xyz'];uv=s['uv'];pd=pixel_descriptors(s['rgb']);labels=s['labels'];scope=request['scope']
 for g,rep in enumerate(a['group_representatives']):
  b=boxes[rep];rect=a['rects'][g].astype(float);size=rect[2:]-rect[:2];valid=bool(a['rect_valid'][g]) and a['project_reliability'][g]>=.99
  if not valid or scope.startswith('UNKNOWN') or min(size)<=0:continue
  if scope!='whole' and size[1]/max(size[0],1)<1.2:continue
  ylo,yhi=(.15,.6) if scope=='upper' else (.45,.95) if scope=='lower' else (0.,1.)
  roi=np.r_[rect[0],rect[1]+size[1]*ylo,rect[2],rect[1]+size[1]*yhi];roi_size=roi[2:]-roi[:2];pixel_ok=(uv[:,0]>=roi[0])&(uv[:,0]<=roi[2])&(uv[:,1]>=roi[1])&(uv[:,1]<=roi[3]);inner=(np.abs(xs-b[:3])<=b[3:]*.85/2).all(1)&pixel_ok;outer=(np.abs(xs-b[:3])<=b[3:]*1.15/2).all(1);base=(np.abs(xs-b[:3])<=b[3:]/2).all(1)&pixel_ok;ii=np.flatnonzero(base);counts[g]=len(ii)
  if len(ii)>=12:
   cov=np.ptp(uv[ii],axis=0)/np.maximum(roi_size,1);box_known[g]=bool((cov>=.35).all());all_rgb[g]=pd[ii].mean(0)
  per=[];perhigh=[];oks=[]
  for k,lab in enumerate(labels):
   core=np.flatnonzero(inner);used=[]
   if len(core)>=12:
    freq=np.bincount(lab[core]);cs=np.flatnonzero((freq>=12)&(freq>=.15*len(core)))
    for c in cs:
     points=np.flatnonzero(lab==c);pure=float(outer[points].mean());take=points[pixel_ok[points]]
     if pure>=.8 and len(take)>=12:used.extend(take.tolist())
   used=np.unique(used);components[k,g]=len(np.unique(lab[used])) if len(used) else 0
   if len(used)>=12 and len(core) and inner[used].sum()>=.7*len(core):
    cov=np.ptp(uv[used],axis=0)/np.maximum(roi_size,1);coverage[k,g]=float(min(cov));oks.append(bool((cov>=.35).all()));summ=np.stack([pd[used[lab[used]==c]].mean(0) for c in np.unique(lab[used])]);per.append(summ.min(0));perhigh.append(summ.max(0))
   else:oks.append(False);per.append(np.zeros(13));perhigh.append(np.zeros(13))
  if all(oks):known[g]=True;desc[g]=np.minimum(per[0],per[1]);high[g]=np.maximum(perhigh[0],perhigh[1])
 return {'known':known,'descriptor':desc,'high_descriptor':high,'box_known':box_known,'box_descriptor':all_rgb,'counts':counts,'component_counts':components,'coverage':coverage}
