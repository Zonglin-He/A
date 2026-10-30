"""Non-learning R7 RGB-text-point pseudo center; pure NumPy, no data/model I/O."""
from dataclasses import dataclass,asdict
import itertools,re
import numpy as np

COLORS=('red','green','blue','yellow','white','black')
COLOR_CYCLE=dict(zip(COLORS,('green','blue','yellow','red','black','white')))
NOUNS=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
@dataclass(frozen=True)
class GConfig:
    crop_margin_m:float=.25
    component_radius_m:float=.35
    min_component_points:int=12
    component_dominance_ratio:float=1.5
    duplicate_box_iou:float=.8
    alternative_support_ratio:float=.8
    min_face_points:int=6
    face_band_size_fraction:float=.15
    face_band_cap_m:float=.30
    tangent_span_fraction:float=.4
    axial_span_fraction:float=.8
    face_MAD_max_m:float=.03
    opposing_residual_difference_max_m:float=.04
    delta_size_fraction_max:float=.1
    delta_absolute_max_m:float=.2
    chromatic_saturation_min:float=.45
    chromatic_value_min:float=.25
    white_saturation_max:float=.2
    white_value_min:float=.8
    black_value_max:float=.2
DEFAULT_CONFIG=GConfig()

def target_color(text):
    text=re.split(r'\.\s*not\s+mentioned\b',str(text),maxsplit=1,flags=re.I)[0].strip().lower()
    target=re.split(r'\b(?:behind|beside|between|near|next\s+to|in\s+front\s+of|left\s+of|right\s+of|in|on|at|with|that|which)\b',text,maxsplit=1)[0]
    if re.search(r'\b(?:not|no|without|neither|except|non)\b',target):return None,'negated_target'
    words=re.findall(r'\b(?:'+'|'.join(COLORS)+r')\b',target)
    if len(words)!=1:return None,'unsupported_or_multicolor_target'
    m=re.match(r'^(?:(?:the|a|an)\s+)?('+'|'.join(COLORS)+r')\s+('+'|'.join(NOUNS)+r')\b',target)
    if m is None:return None,'unsupported_opening_phrase'
    return m.group(1),'supported'

def hsv_support(pixels,color,cfg):
    x=np.asarray(pixels,dtype=np.float64)/255.
    hi=x.max(-1);lo=x.min(-1);d=hi-lo;s=np.divide(d,hi,out=np.zeros_like(d),where=hi>0)
    h=np.zeros_like(hi);nz=d>0
    for k in range(3):
        m=nz&(x[:,k]==hi)
        if k==0:v=np.remainder(np.divide(x[:,1]-x[:,2],d,out=np.zeros_like(d),where=nz),6)
        elif k==1:v=np.divide(x[:,2]-x[:,0],d,out=np.zeros_like(d),where=nz)+2
        else:v=np.divide(x[:,0]-x[:,1],d,out=np.zeros_like(d),where=nz)+4
        h[m]=v[m]/6.
    if color=='white':return (s<=cfg.white_saturation_max)&(hi>=cfg.white_value_min)
    if color=='black':return hi<=cfg.black_value_max
    bands={'red':((0.,1/24),(23/24,1.)),'yellow':((1/8,5/24),),'green':((1/4,5/12),),'blue':((7/12,3/4),)}
    hue=np.zeros(len(x),bool)
    for a,b in bands[color]:hue|=(h>=a)&(h<=b)
    return hue&(s>=cfg.chromatic_saturation_min)&(hi>=cfg.chromatic_value_min)

def visible_points(points,rgb,uv,depth):
    """One nearest-depth point per actual image pixel; no source-crop occlusion leak."""
    H,W=rgb.shape[:2];valid=np.isfinite(points).all(1)&np.isfinite(uv).all(1)&np.isfinite(depth)&(depth>0)
    # Continuous FOV is checked before binning: negative rays cannot round in.
    # Half-open pixel cells use floor; right/bottom in-FOV rays stay in bounds.
    valid&=(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    pix=np.zeros_like(uv,dtype=np.int64);pix[valid]=np.floor(uv[valid]).astype(np.int64)
    valid&=(pix[:,0]>=0)&(pix[:,0]<W)&(pix[:,1]>=0)&(pix[:,1]<H)
    ids=np.flatnonzero(valid)
    if not len(ids):return np.empty((0,3)),np.empty((0,3),dtype=np.uint8)
    p=points[ids];key=pix[ids,1]*W+pix[ids,0]
    order=np.lexsort((p[:,2],p[:,1],p[:,0],depth[ids],key));ids=ids[order];keys=key[order]
    ids=ids[np.r_[True,keys[1:]!=keys[:-1]]]
    return points[ids],rgb[pix[ids,1],pix[ids,0],:3]

def components(points,radius):
    p=np.unique(points,axis=0)
    if not len(p):return []
    cell=np.floor(p/radius).astype(np.int64);buckets={}
    for i,c in enumerate(cell):buckets.setdefault(tuple(c),[]).append(i)
    unseen=set(range(len(p)));out=[]
    while unseen:
        first=min(unseen);unseen.remove(first);stack=[first];members=[]
        while stack:
            i=stack.pop();members.append(i)
            for offset in itertools.product((-1,0,1),repeat=3):
                for j in buckets.get(tuple(cell[i]+offset),[]):
                    if j in unseen and np.linalg.norm(p[i]-p[j])<=radius:
                        unseen.remove(j);stack.append(j)
        out.append(p[sorted(members)])
    return sorted(out,key=lambda x:(-len(x),tuple(x[0])))

def aabb_iou(a,b):
    inter=np.maximum(0,np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)-np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2)).prod()
    return float(inter/(a[3:].prod()+b[3:].prod()-inter))

def in_box(p,b,margin):return (np.abs(p-b[:3])<=b[3:]/2+margin).all(1)

def generate_pseudo(*,points,rgb,text,source_boxes,text_scores,query_index,uv,depth,mode='rgb_text',config=DEFAULT_CONFIG):
    """Finite keyword-only allowlist: GT/positive_map/extra fields cause TypeError.
    Returns local target6/w6 + scalar diagnostics, never changes source query/size.
    Modes rgb_text / pc_only / shuffled_color use exactly the same geometry rules.
    """
    assert mode in ('rgb_text','pc_only','shuffled_color')
    p=np.asarray(points,dtype=np.float64);im=np.asarray(rgb);boxes=np.asarray(source_boxes,dtype=np.float64);scores=np.asarray(text_scores,dtype=np.float64);u=np.asarray(uv,dtype=np.float64);z=np.asarray(depth,dtype=np.float64)
    assert p.ndim==2 and p.shape[1]==3 and u.shape==(len(p),2) and z.shape==(len(p),)
    assert im.ndim==3 and im.shape[2]>=3 and im.dtype==np.uint8
    assert boxes.ndim==2 and boxes.shape[1]==6 and np.isfinite(boxes).all()
    assert scores.shape==(len(boxes),) and np.isfinite(scores).all()
    q=int(query_index);assert q==int(np.argmax(scores)), 'query must be fixed pure-text source Top1'
    source=boxes[q].copy();target=source.copy();w=np.zeros(6)
    result={'query_index':q,'target6':target,'reliability6':w,'mode':mode,'size_prior_unchanged':True,'yaw':0.,'diagnostics':{'config':asdict(config),'axis':[]}}
    def skip(reason):result['diagnostics']['reason']=reason;return result
    if not (source[3:]>0).all():return skip('nonpositive_source_size')
    color,reason=target_color(text)
    if color is None:return skip(reason)
    requested=color; color=COLOR_CYCLE[color] if mode=='shuffled_color' else color
    vp,pixels=visible_points(p,im,u,z)
    support=np.ones(len(vp),bool) if mode=='pc_only' else hsv_support(pixels,color,config)
    sp=vp[support];crop=sp[in_box(sp,source,config.crop_margin_m)]
    comps=components(crop,config.component_radius_m)
    result['diagnostics'].update(requested_color=requested,effective_color=color,visible_point_count=len(vp),supported_crop_unique_point_count=len(np.unique(crop,axis=0)),component_sizes=[len(c) for c in comps])
    if not comps or len(comps[0])<config.min_component_points:return skip('insufficient_supported_component')
    if len(comps)>1 and len(comps[0])<config.component_dominance_ratio*len(comps[1]):return skip('ambiguous_components')
    c=comps[0]
    for j,b in enumerate(boxes):
        if j==q or (b[3:]<=0).any() or aabb_iou(source,b)>=config.duplicate_box_iou:continue
        alt=np.unique(sp[in_box(sp,b,config.crop_margin_m)],axis=0)
        if len(alt)>=config.min_component_points and len(alt)>=config.alternative_support_ratio*len(c):
            result['diagnostics']['ambiguous_other_query']=j
            return skip('same_evidence_competing_candidate')
    for axis in range(3):
        tangent=[i for i in range(3) if i!=axis];size=source[3+axis]
        band=min(config.face_band_size_fraction*size,config.face_band_cap_m)
        faces=[];a={'axis':axis,'accepted':False,'face_counts':[]}
        for sign in (-1,1):
            location=source[axis]+sign*size/2
            f=c[np.abs(c[:,axis]-location)<=band];a['face_counts'].append(len(f))
            if len(f)<config.min_face_points:continue
            spans=np.ptp(f[:,tangent],axis=0)
            if np.any(spans<config.tangent_span_fraction*source[3+np.array(tangent)]):continue
            residual=f[:,axis]-location;median=float(np.median(residual));mad=float(np.median(np.abs(residual-median)))
            if mad>config.face_MAD_max_m:continue
            faces.append((median,mad))
        if len(faces)==2 and np.ptp(c[:,axis])>=config.axial_span_fraction*size:
            delta=(faces[0][0]+faces[1][0])/2;limit=min(config.delta_size_fraction_max*size,config.delta_absolute_max_m)
            a.update(opposing_median_residuals=[f[0] for f in faces],opposing_MAD=[f[1] for f in faces],delta=delta,delta_limit=limit)
            if abs(faces[0][0]-faces[1][0])<=config.opposing_residual_difference_max_m and abs(delta)<=limit:
                target[axis]+=delta;w[axis]=1.;a['accepted']=True
        result['diagnostics']['axis'].append(a)
    return skip('accepted_some_center_axes' if w.any() else 'no_two_sided_reliable_axis')
