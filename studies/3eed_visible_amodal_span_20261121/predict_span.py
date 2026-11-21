"""Seal the physical predictions before scoring. No task-label or oracle imports."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from common import BASE, HERE, OUT, PRIVATE, QWEN, ROOT, SPEC, digest, inputs, read, write
from geometry import camera, raw_returns, rigid_xyz
from span import TOL, axes, center_interval, offset, read_endpoint
sys.path[:0]=[str(BASE),str(BASE.parent/'3eed_qwen_four_scale_region_grounding_20261121')]
from region_depth import (held_cells, inside, model_sample, model_to_camera_matrix,
                          partition, transfer, z_buffer)
from predict import to_raw_support

ARMS=('frozen','real','anchor','copy','pair','wrong','point')

def raw_context(row,entry):
    spec,size,mat=camera(row)
    uv,cam=raw_returns(row,spec,size)
    win=np.array(entry['pixel'])*np.tile(size,2)/1000.
    # Include nearest returns from whole z-buffer cells at the window edges.
    ids,cells=z_buffer(uv,cam,np.arange(len(uv)))
    active=inside(uv[ids],win)
    uv,cam,held=uv[ids[active]],cam[ids[active]],held_cells(cells[active])
    sample=model_sample(row)
    cuts,labels,births,_=partition(sample)
    node=entry['node'];level=births[node]
    support=sample[labels[level]==node]
    np.testing.assert_allclose(to_raw_support(support,row['target'],row['suffix']),entry['support_raw'],atol=1.e-9,rtol=0.)
    mxyz=rigid_xyz(cam,np.linalg.inv(model_to_camera_matrix(spec)))
    member=transfer(sample,mxyz,cuts,labels,births)[level]==node
    # Identify actual source draws by their native raw coordinates, also
    # excluding co-located duplicates from the independent witness subset.
    path=row['points']
    raw=np.load(path,mmap_mode='r',allow_pickle=False) if path.endswith('.npy') else np.fromfile(path,'<f4').reshape(-1,4)
    ix=np.random.default_rng(row['seed']).choice(len(raw),16384,replace=len(raw)<16384)
    xyz=rigid_xyz(cam,np.linalg.inv(mat))
    sampled=cKDTree(raw[ix,:3]).query(xyz)[0]<1.e-5
    return spec,mat,uv,cam,xyz,held,member,sampled

def nearest(pixel,uv,allowed):
    ids=np.flatnonzero(allowed)
    if not len(ids):
        return None,np.inf
    err=np.linalg.norm(uv[ids]-pixel,axis=1)
    j=int(np.argmin(err))
    return (int(ids[j]),float(err[j])) if err[j]<=4. else (None,float(err[j]))

def one_read(entry,obs,parsed,crop,context=None):
    result=dict(offsets=dict.fromkeys(ARMS,0.),selected=entry['selected'],status='upstream_fallback',
                read_status=parsed['status'])
    if not entry['selected']:
        return result
    box=np.array(entry['boxes']['H'])
    normal,width=axes(box)
    xyz=np.asarray(entry['support_raw'])
    values=(xyz-box[:3])@normal
    a,b=float(values.min()),float(values.max())
    real=center_interval(a,b,width)
    result.update(status='no_rgb_pair',real_span_m=b-a,width_m=width,real_interval=real)
    if real is None:
        result['status']='real_extent_infeasible'
        return result
    dx=result['offsets']
    for arm in ('real','anchor','copy','pair','wrong','point'):
        dx[arm]=offset(real)
    if parsed['status']!='pair':
        return result
    spec,mat,uv,camera_points,raw,held,member,sampled=context
    crop=np.asarray(crop)
    pix={k:crop[:2]+np.asarray(parsed[k])*(crop[2:]-crop[:2])/1000. for k in
         ('left','right','other_left','other_right') if parsed.get(k) is not None}
    look={k:nearest(pix[k],uv,member&~held) for k in ('left','right')}
    side=min(look,key=lambda k:look[k][1] if look[k][0] is not None else np.inf)
    aid,dist=look[side]
    result['part']=parsed['part']
    if aid is None:
        result['status']='no_anchor_return'
        return result
    other='right' if side=='left' else 'left'
    p=raw[aid]
    station=float((p-box[:3])@normal)
    a,b=min(a,station),max(b,station)
    anchored=center_interval(a,b,width)
    result.update(anchor_side=side,anchor_px=dist,anchor_interval=anchored,
                  anchor_range=[a,b],physical_anchor=p.tolist(),observed_uv=uv[aid].tolist(),
                  opposite_uv=pix[other].tolist(),part_pixels=[pix['left'].tolist(),pix['right'].tolist()])
    if anchored is None:
        result['status']='anchor_extent_infeasible'
        return result
    for arm in ('anchor','copy','pair','wrong','point'):
        dx[arm]=offset(anchored)
    nc=mat[:3,:3]@normal
    for arm,field in [('pair',other),('wrong','other_'+other)]:
        if field not in pix:
            continue
        if arm=='wrong' and np.linalg.norm(pix[field]-pix[other])<=8:
            continue
        e=read_endpoint(camera_points[aid],nc,station,pix[field],spec,mat,normal,width)
        result[arm+'_read']=e
        if arm=='pair':
            result['status']=e['status']
            i=center_interval(a,b,width,e['copy'])
            if i is not None:
                dx['copy']=offset(i)
            if 'nominal_s' in e and e.get('reprojection_px',np.inf)<=4.:
                pt=station+e['nominal_s']
                i=center_interval(a,b,width,(pt,pt))
                if i is not None:
                    dx['point']=offset(i)
        if e['status']=='feasible':
            interval=center_interval(a,b,width,e['endpoint'])
            if interval is not None:
                dx[arm]=offset(interval)
                if arm=='pair':
                    result.update(status='applied_or_already_inside',interval=interval,
                                  new_span_m=max(0.,e['endpoint'][0]-b,a-e['endpoint'][1]))
            elif arm=='pair':
                result['status']='combined_extent_infeasible'
    wid,error=nearest(pix[other],uv,held)
    if wid is not None:
        result['witness']=dict(z=float(camera_points[wid,2]),pixel_error=error,
                in_selected_region=bool(member[wid]),not_in_frozen_tree_draws=bool(not sampled[wid]),
                xyz=raw[wid].tolist(),uv=uv[wid].tolist())
    return result

def process(item):
    row,entry,reply,manual=item
    parse=reply['parsed'] if reply else {'status':'not_queried'}
    crop=reply['crop'] if reply else None
    context=None
    if parse['status']=='pair' or (manual and manual.get('status')=='pair'):
        assert digest(row['points'])==row['point_sha']
        context=raw_context(row,entry)
    res=one_read(entry,row,parse,crop,context)
    if manual is not None:
        res['independent']=one_read(entry,row,manual,crop,context)
    return res

def run():
    started=perf_counter()
    panel, frozen=inputs()
    landmark=read(OUT/'landmark_seal.json')
    hashes=read(PRIVATE/'reply_hashes.json')
    assert landmark['sha256']==digest(PRIVATE/'reply_hashes.json')
    assert landmark['pins']['protocol']==digest(SPEC) and landmark['pins']['code']==digest(HERE/'landmarks.py')
    review=read(OUT/'review_seal.json')
    assert review['marks_sha']==digest(PRIVATE/'independent_marks.json') and review['roster_sha']==digest(PRIVATE/'review_roster.json')
    marks=read(PRIVATE/'independent_marks.json')
    manual={v['uid']:marks[k] for k,v in read(PRIVATE/'review_roster.json').items()}
    tasks=[]
    for row in panel:
        uid=row['uid'];entry=frozen[uid];reply=None
        if entry['selected']:
            path=PRIVATE/'replies'/(uid+'.json')
            assert digest(path)==hashes[uid]
            reply=read(path)
            assert reply['pin']==landmark['pins'] and reply['uid']==uid
        tasks.append((row,entry,reply,manual.get(uid)))
    with ProcessPoolExecutor(max_workers=5) as pool:
        results={r['uid']:v for r,v in zip(panel,pool.map(process,tasks,chunksize=4),strict=True)}
    write(PRIVATE/'predictions.json',results)
    seal=dict(frames=len(results),private_sha=digest(PRIVATE/'predictions.json'),
              source=digest(BASE/'results/prediction_seal.json'),landmarks=digest(OUT/'landmark_seal.json'),
              review=digest(OUT/'review_seal.json'),protocol=digest(SPEC),
              implementation={f:digest(HERE/f) for f in ('common.py','span.py','predict_span.py')},
              stages=dict(Counter(v['status'] for v in results.values())),seconds=perf_counter()-started,
              source_training=0,target_updates=0,target_labels_read=False,FINAL_access=False)
    write(OUT/'prediction_seal.json',seal)
    print(seal,flush=True)

if __name__=='__main__':
    run()
