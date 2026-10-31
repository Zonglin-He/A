"""R7 finite input/output engineering. No annotation-label loader or model I/O."""
from pathlib import Path
import hashlib,json,re
import numpy as np
INPUT_KEYS=frozenset({'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'})
MODES=('rgb_text','pc_only','shuffled_color')
KINDS=('direct_G','param1','param3','outputspace1','outputspace3')
ARM_NAMES=('source',)+tuple(m+'__'+k for m in MODES for k in KINDS)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def validate_manifest(rows):
    assert rows and len({r['id'] for r in rows})==len(rows)
    for r in rows:
        assert set(r)==INPUT_KEYS and r['platform'] in ('drone','quad')
        assert re.fullmatch(r'[A-Za-z0-9_-]+',r['id']) and isinstance(r['caption'],str) and r['caption']
        assert isinstance(r['seed'],int) and not isinstance(r['seed'],bool)
        for key in ('pointpath','metapath','imagepath'):assert Path(r[key]).is_file()
    return rows

def sample_points(row):
    p=Path(row['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
    assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
    # R5 exact convention: raw float32 and in-place float32 Drone shift.
    raw=np.asarray(raw,np.float32);ix=np.random.default_rng(row['seed']).choice(len(raw),16384,replace=len(raw)<16384)
    xyz=raw[ix,:3].copy()
    if row['platform']=='drone':xyz[:,2]+=1.8
    return np.ascontiguousarray(xyz,np.float32),ix.astype(np.int64),len(raw)

def eval_boxes(raw):
    b=np.array(raw,dtype=np.float32,copy=True);b[...,3:6]=np.maximum(b[...,3:6],np.float32(1e-6));return b

def npz_payload(*,sample_id,source_boxes,scores,native_top10,q,selected_arms,statuses,pseudo_targets,pseudo_weights,point_sha,image_sha):
    source=np.asarray(source_boxes,np.float32);scores=np.asarray(scores,np.float32);arms=np.asarray(selected_arms,np.float32)
    assert source.shape==(256,6) and scores.shape==(256,) and arms.shape==(16,6)
    assert len(statuses)==16 and np.array_equal(arms[0],source[q])
    targets=np.asarray(pseudo_targets,np.float32);weights=np.asarray(pseudo_weights,np.float32)
    assert targets.shape==weights.shape==(3,6) and ((weights>=0)&(weights<=1)).all()
    assert not weights[:,3:].any() and np.array_equal(targets[:,3:],np.tile(source[q,3:],(3,1)))
    assert np.isfinite(source).all() and np.isfinite(scores).all() and np.isfinite(arms).all()
    top=np.asarray(native_top10,np.int64);assert top.shape==(10,) and len(np.unique(top))==10
    assert q==int(np.argmax(scores));assert np.all(scores[top[:-1]]>=scores[top[1:]])
    return {'sample_id':np.array(sample_id),'source_boxes_raw6':source,'source_boxes_eval6':eval_boxes(source),'source_text_scores256':scores,'source_native_top10':top,'fixed_query_index':np.array(q,np.int64),'native_top1_differs_fixed_query':np.array(int(top[0])!=q),'arm_names':np.array(ARM_NAMES),'arm_modes':np.array(('source',)+tuple(m for m in MODES for k in KINDS)),'arm_status':np.array(statuses),'arm_boxes_raw6':arms,'arm_boxes_eval6':eval_boxes(arms),'mode_names':np.array(MODES),'pseudo_target6':targets,'pseudo_reliability6':weights,'point_sample_sha256':np.frombuffer(bytes.fromhex(point_sha),np.uint8),'image_file_sha256':np.frombuffer(bytes.fromhex(image_sha),np.uint8)}

def to_json(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):to_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [to_json(v) for v in x]
    return x
