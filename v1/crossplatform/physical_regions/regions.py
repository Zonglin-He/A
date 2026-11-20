"""Construct nested physical regions in the model coordinate frame."""
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import softmax
from crossplatform.candidate_frontend import WAYMO_VIEWS
from crossplatform.round7.camera_projection import CameraSpec, read_camera_allowlist, project_model_points, model_to_camera_matrix
from .language import CLASSES

# Indices on the original eight-quantile grid. Keeping the grid fixes the radii.
FOUR_LEVEL_INDICES = (0, 2, 5, 7)


def camera(platform, index, calibration, image):
    spec = CameraSpec.from_camera_fields(platform, read_camera_allowlist(calibration),
                                         waymo_view_suffix=index, missing_distortion_policy='assume_zero_declared')
    with Image.open(image) as photo:
        return spec, photo.size


CORNERS = np.array([(x,y,z) for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])

def box_projection(boxes, target, view, calib, image):
    spec, size = camera(target, int(view or 0), str(calib), str(image))
    boxes = np.asarray(boxes)
    corners = boxes[:,None,3:6] * CORNERS
    x, y = corners[...,0].copy(), corners[...,1].copy()
    co, si = np.cos(boxes[:,6,None]), np.sin(boxes[:,6,None])
    corners[...,0],corners[...,1] = co*x-si*y, si*x+co*y
    center = project_model_points(boxes[:,:3],spec,size)
    verts = project_model_points((corners+boxes[:,None,:3]).reshape(-1,3),spec,size)
    img = verts['uv'][:,0].reshape(-1,8) / size[0]
    depth = verts['depth'].reshape(-1,8)
    radius = np.linalg.norm((np.c_[boxes[:,:3],np.ones(len(boxes))] @ model_to_camera_matrix(spec).T)[:,:3],axis=1)
    return np.column_stack((center['uv'][:,0]/size[0], center['depth'], img.min(1), img.max(1),
                            depth.min(1), depth.max(1), radius,
                            center['in_image'] & np.isfinite(img).all(1) & (depth.min(1)>.1)))


def axes(box):
    c,s=np.cos(box[6]),np.sin(box[6])
    return np.array([[c,s,0],[-s,c,0],[0,0,1.]])

class Returns:
    def __init__(self, points):
        self.xyz=np.unique(points.astype(float),axis=0)
        self.tree=cKDTree(self.xyz)

    def local(self, box):
        half=box[3:6]*.5
        ids=np.asarray(self.tree.query_ball_point(box[:3],np.linalg.norm(half)+1.e-7),int)
        off=(self.xyz[ids]-box[:3]) @ axes(box).T
        inside=(abs(off)<=half+1.e-7).all(1)
        return ids[inside],off[inside]

def lexicon(xyz, scores):
    seeds=Returns(xyz)
    _,inv=np.unique(xyz,axis=0,return_inverse=True)
    centered=scores.astype(float)-scores.mean(-1,keepdims=True,dtype=float)
    count=np.bincount(inv)
    averaged=[]
    for matrix in (centered.mean(0),centered[0]):
        out=np.zeros((len(count),matrix.shape[1]))
        np.add.at(out,inv,matrix)
        averaged.append(out/count[:,None])
    return seeds,averaged[0],averaged[1]


def terrain(xyz):
    cell=np.floor(xyz[:,:2]).astype(int)
    order=np.lexsort((xyz[:,2],cell[:,1],cell[:,0]))
    x=xyz[order[np.r_[True,(np.diff(cell[order],axis=0)!=0).any(1)]]]
    if len(x)<12:
        return np.full(3,np.nan)
    design=np.column_stack((x[:,:2],np.ones(len(x))))
    rng=np.random.default_rng(5927)
    slope=np.tan(np.deg2rad(25.))
    best=(-1,-np.inf)
    fit=None
    for _ in range(96):
        ids=rng.choice(len(x),3,replace=False)
        try:
            v=np.linalg.solve(design[ids],x[ids,2])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(v[:2])>slope:continue
        error=np.abs(x[:,2]-design@v)
        hit=error<=.20
        if hit.sum()<12:continue
        rank=(hit.sum(),-np.median(error[hit]))
        if rank>best:best=rank;fit=hit
    if fit is None:return np.full(3,np.nan)
    v,_,rank,_=np.linalg.lstsq(design[fit],x[fit,2],rcond=None)
    return v if rank==3 and np.linalg.norm(v[:2])<=slope else np.full(3,np.nan)

def projection(x,f):
    view=WAYMO_VIEWS.index(f['waymo_view'])
    cam,size=camera(f['target'],view,str(f['calibration']),str(f['rgb']))
    p=project_model_points(x,cam,size)
    valid=p['in_image'];uv=np.where(valid[:,None],p['uv'],0.)
    nx=(size[0]+3)//4
    tile=np.floor(uv/4).astype(int);flat=tile[:,1]*nx+tile[:,0]
    zbuf=np.full(nx*((size[1]+3)//4),np.inf)
    np.minimum.at(zbuf,flat[valid],p['depth'][valid])
    seen=valid & (p['depth']<=zbuf[flat]+.30+.02*p['depth'])
    return uv[:,0]/size[0],p['depth'],seen

def levels(x,fg,indices=FOUR_LEVEL_INDICES):
    cuts=np.full(len(indices),np.nan)
    out=np.full((len(indices),len(x)),-1,dtype=np.int32)
    ids=np.flatnonzero(fg)
    if len(ids)<9:return cuts,out
    dist,neigh=cKDTree(x[ids]).query(x[ids],k=9)
    u=np.repeat(np.arange(len(ids)),8)
    v=neigh[:,1:].reshape(-1)
    d=np.maximum.reduce((dist[:,1:].reshape(-1),dist[u,4],dist[v,4]))
    cuts=np.quantile(d,(np.asarray(indices)+.5)/8)
    for k,cut in enumerate(cuts):
        mask=d<=cut
        graph=coo_matrix((np.ones(mask.sum(),np.uint8),(u[mask],v[mask])),shape=(len(ids),len(ids))).tocsr()
        count,lab=connected_components(graph,directed=False)
        keep=np.bincount(lab,minlength=count)>=8
        lookup=np.where(keep,np.cumsum(keep)-1,-1)
        out[k,ids]=lookup[lab]
    return cuts,out

def describe(x,labels,pointscores,proj):
    u,depth,view=proj
    count=int(labels.max())+1
    sizes=np.bincount(labels[labels>=0],minlength=count)
    scores=np.zeros((count,pointscores.shape[1]))
    for k in range(scores.shape[1]):
        scores[:,k]=np.bincount(labels[labels>=0],weights=pointscores[labels>=0,k],minlength=count)/sizes
    desc=np.full((count,5),np.nan)
    ids=np.flatnonzero(view & (labels>=0))
    if len(ids):
        order=ids[np.argsort(labels[ids],kind='stable')]
        groups=np.split(order,np.flatnonzero(np.diff(labels[order]))+1)
        for ix in groups:
            g=int(labels[ix[0]])
            lo,mid,hi=np.quantile(u[ix],[.25,.5,.75])
            desc[g]=[lo,mid,hi,np.median(depth[ix]),len(ix)]
    return sizes.astype(np.int32),softmax(scores,axis=1),desc

def build(x,scores,seeds,frame,indices=FOUR_LEVEL_INDICES):
    xyz=np.unique(x.astype(float),axis=0)
    ground=terrain(xyz)
    height=xyz[:,2]-xyz[:,:2]@ground[:2]-ground[2]
    cuts,lab=levels(xyz,height>.20,indices)
    field,centered,_=lexicon(seeds,scores)
    centered-=centered.mean(0,keepdims=True)
    point_scores=centered[field.tree.query(xyz)[1]]
    proj=projection(xyz,frame)
    result=dict(xyz=xyz.astype(np.float32),plane=ground,labels=lab,cuts=cuts)
    for k in range(len(cuts)):
        size,prob,desc=describe(xyz,lab[k],point_scores,proj)
        result['size'+str(k)]=size;result['p'+str(k)]=prob;result['view'+str(k)]=desc
    return result


def split_returns(x):
    values=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits=values[:,0]*np.uint64(73856093)^values[:,1]*np.uint64(19349663)^values[:,2]*np.uint64(83492791)
    bits^=bits>>np.uint64(16)
    bits*=np.uint64(0x45d9f3b)
    bits^=bits>>np.uint64(16)
    return (bits & 1).astype(bool)

def bottom(box,x,ground,split):
    initial=box[2]-box[5]/2
    if not np.isfinite(ground).all():return initial
    off=x[:,:2]-box[:2]
    local=off@axes(box)[:2,:2].T
    half=box[3:5]/2
    ring=(abs(local)>half).any(1) & (abs(local)<=2*half).all(1)
    terrain=abs(x[:,2]-x[:,:2]@ground[:2]-ground[2])<=.20
    estimates=[]
    for side in (False,True):
        mask=ring&terrain&(split==side)
        if mask.sum()<3:return initial
        xy=off[mask]
        angles=np.sort(np.arctan2(xy[:,1],xy[:,0]))
        if np.diff(np.r_[angles,angles[0]+2*np.pi]).max()>=np.pi:return initial
        v,_,rank,_=np.linalg.lstsq(np.c_[xy,np.ones(len(xy))],x[mask,2],rcond=None)
        if rank<3 or np.linalg.norm(v[:2])>np.tan(np.deg2rad(25.)):return initial
        estimates.append(v[2])
    value=np.clip(initial,min(estimates),max(estimates))
    return value if value<box[2]+box[5]/2 else initial

def only_ground(anchor,z):
    b=anchor.copy();top=anchor[2]+anchor[5]/2
    b[2]=(top+z)/2;b[5]=top-z
    return b


def measured_box(points,plane):
    mid=points[:,:2].mean(0)
    xy=points[:,:2]-mid
    _,directions=np.linalg.eigh(xy.T@xy)
    long=directions[:,-1]
    yaw=float(np.arctan2(long[1],long[0]) % np.pi)
    axes=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    local=xy@axes.T
    lo,hi=local.min(0),local.max(0)
    center=(lo+hi)/2@axes+mid
    low,top=np.min(points[:,2]),np.max(points[:,2])
    if np.isfinite(plane).all():
        low=min(low,center@plane[:2]+plane[2])
    b=np.array([*center,(top+low)/2,*(hi-lo),top-low,yaw])
    b[3:6]=np.maximum(b[3:6],1e-6)
    return b

def complete(box,dimensions):
    b=box.copy()
    yaw=b[6]
    axes=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    dims=dimensions.copy()
    dims[:2]=np.sort(dims[:2])[::-1] if b[3]>=b[4] else np.sort(dims[:2])
    slab=b[:2]@axes.T
    advance=np.where(np.abs(slab)>b[3:5]/2,np.sign(slab),0.)
    increase=np.maximum(dims-b[3:6],0.)
    b[:2]+=(advance*increase[:2]/2)@axes
    b[2]+=increase[2]/2
    b[3:6]+=increase
    return b

def build_inventory(seg,word):
    x=seg['xyz'].astype(float)
    loc,inv=np.unique(word['xyz'],return_inverse=True,axis=0)
    pred=np.maximum(word['extent'].astype(float),1e-6)
    dim=np.zeros((len(loc),)+pred.shape[1:])
    np.add.at(dim,inv,pred)
    dim/=np.bincount(inv)[:,None,None]
    near=cKDTree(loc).query(x)[1]
    allids=np.full_like(seg['labels'],-1)
    index={};para=[];levels=[];size=[];boxes=[];views=[];probs=[];stops=[];dimensions=[]
    for k,lab in enumerate(seg['labels']):
        ordered=np.flatnonzero(lab>=0)
        if not len(ordered):continue
        ordered=ordered[np.argsort(lab[ordered],kind='stable')]
        for ids in np.split(ordered,np.flatnonzero(np.diff(lab[ordered]))+1):
            key=int(ids.min()),len(ids)
            if key not in index:
                n=len(para);index[key]=n
                c=int(lab[ids[0]])
                box=measured_box(x[ids],seg['plane'])
                proto=np.median(dim[np.unique(near[ids])],axis=0)
                obs=box[3:6].copy();obs[:2]=np.sort(obs[:2])[::-1]
                canonical=proto.copy();canonical[:,:2]=np.sort(proto[:,:2],axis=1)[:,::-1]
                match=np.prod(np.minimum(obs,canonical)/np.maximum(obs,canonical),axis=1)
                s=seg['p'+str(k)][c,:-1]*match
                probs.append(s/max(s.sum(),np.finfo(float).tiny))
                stops.append(match.max());para.append(-1);levels.append(0)
                size.append(len(ids));boxes.append(box);views.append(seg['view'+str(k)][c])
                dimensions.append(proto)
            n=index[key];levels[n]|=1<<k
            allids[k,ids]=n
            if k:
                for child in np.unique(allids[k-1,ids]):
                    if child>=0 and child!=n:
                        assert child<n and para[child] in (-1,n)
                        para[child]=n
    count=len(para);nclass=len(CLASSES)-1
    return dict(parent=np.array(para,int),level=np.array(levels,int),size=np.array(size,int),
        boxes=np.array(boxes,float).reshape(count,7),dimensions=np.array(dimensions,float).reshape(count,nclass,3),
        view=np.array(views,float).reshape(count,5),p=np.array(probs,float).reshape(count,nclass),
        stop=np.array(stops),assign=allids,xyz=seg['xyz'],plane=seg['plane'])


def assemble(seg,word):
    """Materialize all noun-conditioned boxes for callers that inspect both endpoints."""
    data = build_inventory(seg,word)
    sizes = data.pop('dimensions')
    boxes = [[complete(box,d) for d in row] for box,row in zip(data['boxes'],sizes)]
    data['completed'] = np.asarray(boxes,dtype=float).reshape(*sizes.shape[:2],7)
    return data
