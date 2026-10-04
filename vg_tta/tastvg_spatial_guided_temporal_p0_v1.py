"""Pure geometry for the three-arm target-centric temporal-input audit."""
import hashlib, json
import numpy as np

ARMS = ['Full', 'A_ROI', 'GT_ROI']
FIELDS = [a+'_t' for a in ARMS] + ['A_ROI_minus_Full_t', 'GT_ROI_minus_Full_t',
    'GT_ROI_minus_A_ROI_t'] + [a+'_success' for a in ARMS] + [a+'_disjoint' for a in ARMS]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def a_xyxy(boxes, width, height):
    b=np.asarray(boxes, dtype=float)
    assert b.ndim==2 and b.shape[1]==4
    return np.concatenate((b[:,:2]-b[:,2:]/2, b[:,:2]+b[:,2:]/2),1)*[width,height,width,height]

def extend_gt(truth, frame_ids):
    """Spatial interpolation only. Never accepts a temporal event interval."""
    keys=sorted(map(int, truth)); assert keys
    b=np.array([truth[str(i)] if str(i) in truth else truth[i] for i in keys], float)
    assert b.shape==(len(keys),4) and np.isfinite(b).all()
    assert np.all(b[:,2:]>b[:,:2])
    out=np.stack([np.interp(frame_ids, keys, b[:,j]) for j in range(4)],1)
    return out, dict(annotation_frames=len(keys), extrapolated_frames=sum(i<keys[0] or i>keys[-1] for i in frame_ids),
        sampled_frames=len(frame_ids), temporal_span_used=False, perfect_tracking_outside_support=False)

def window(box, width, height, context=1.5):
    b=np.asarray(box,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]):
        return dict(fallback_full=True, reason='invalid_spatial_box')
    centre=(b[:2]+b[2:])/2
    side=max(2,int(np.ceil(context*np.max(b[2:]-b[:2]))))
    # The fixed context rule is not retuned or clipped to temporal annotations.
    x,y=np.floor(centre-side/2).astype(int).tolist()
    return dict(fallback_full=False,x=x,y=y,side=side,
        padding=[max(0,-x),max(0,-y),max(0,x+side-width),max(0,y+side-height)])

def crop(frame, spec):
    if spec['fallback_full']: return frame
    x,y,n=spec['x'],spec['y'],spec['side'];h,w=frame.shape[:2]
    # Edge replication preserves the requested centre and square size at borders.
    ix=np.clip(np.arange(x,x+n),0,w-1);iy=np.clip(np.arange(y,y+n),0,h-1)
    return np.ascontiguousarray(frame[iy[:,None],ix[None,:]])

def sample_indices(ids, fps):
    assert len(ids)>1 and ids==sorted(set(ids)) and fps>0
    duration=(ids[-1]-ids[0]+1)/fps
    physical=ids[0]+np.arange(max(1,int(duration*2)))/2*fps
    return np.abs(np.asarray(ids)[None,:]-physical[:,None]).argmin(1),duration

def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    ov=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return float(ov/(a[1]-a[0]+b[1]-b[0]-ov))

def top(proposals, confidence):
    p=np.asarray(proposals,float);c=np.asarray(confidence,float)
    assert p.ndim==2 and p.shape[1]==2 and len(p)==len(c)>0
    assert np.isfinite(p).all() and np.isfinite(c).all() and np.all(p[:,1]>p[:,0])
    k=int(np.argmax(c));return dict(index=k,interval=p[k].tolist(),confidence=float(c[k]),proposal_count=len(p))

def metrics(row, span):
    out={}
    for arm in ARMS:
        t=iou(row['selections'][arm]['interval'],span)
        out.update({arm+'_t':t,arm+'_success':float(t>.5),arm+'_disjoint':float(t==0)})
    for a,b in [('A_ROI','Full'),('GT_ROI','Full'),('GT_ROI','A_ROI')]:out[a+'_minus_'+b+'_t']=out[a+'_t']-out[b+'_t']
    return out

def data_guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        s=str(args[0])
        if any(k in s for k in ['GT_LABELS','GT_EXPOSURE','GT_SUBSET','labels_diagnostic',
            '/GT_BOXES_INPUT.json','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','vidstd-test-anno']):
            raise PermissionError('Deployable Full/A-ROI worker forbids GT and scored results')
