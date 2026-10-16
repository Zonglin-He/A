"""Prospective Source-seeded foreground hypothesis; one matched evidence change."""
from pathlib import Path
import copy,gc,hashlib,io,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,LITE,OLD,read,write,put,sha,verify,guard
SP=BASE/'spatial'

def foreground(rgb,box,seed):
    import cv2,numpy as np
    from scripts.stvg_weightfree_spatial_v1 import bbox
    h,w=rgb.shape[:2];a,b,c,d=[float(x)for x in box];a,b,c,d=max(0,a),max(0,b),min(1,c),min(1,d)
    if c<=a or d<=b:return None,dict(status='invalid_source_intersection')
    cx,cy=(a+c)/2,(b+d)/2;ew,eh=(c-a)*1.25,(d-b)*1.25
    x0,y0=max(0,int((cx-ew/2)*w)),max(0,int((cy-eh/2)*h));x1,y1=min(w,int(np.ceil((cx+ew/2)*w))),min(h,int(np.ceil((cy+eh/2)*h)))
    mask=np.zeros((h,w),np.uint8);mask[y0:y1,x0:x1]=cv2.GC_PR_BGD
    sx0,sy0=max(0,int(a*w)),max(0,int(b*h));sx1,sy1=min(w,int(np.ceil(c*w))),min(h,int(np.ceil(d*h)));mask[sy0:sy1,sx0:sx1]=cv2.GC_PR_FGD
    yy,xx=np.mgrid[:h,:w];core=((xx+.5-cx*w)/max(1,(c-a)*w*.25))**2+((yy+.5-cy*h)/max(1,(d-b)*h*.25))**2<=1
    mask[core]=cv2.GC_FGD;mask[[0,-1],:]=cv2.GC_BGD;mask[:,[0,-1]]=cv2.GC_BGD;initial=mask.copy()
    if not np.any(mask==cv2.GC_FGD)or not np.any(mask==cv2.GC_BGD):return None,dict(status='insufficient_foreground_background_support')
    bg=np.zeros((1,65),np.float64);fg=bg.copy();cv2.setRNGSeed(int(seed)%2147483647)
    try:cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),mask,None,bg,fg,3,cv2.GC_INIT_WITH_MASK)
    except cv2.error:return None,dict(status='opencv_grabcut_failed',GT_read=False)
    full=(mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD);out=bbox(full)
    if out is None:return None,dict(status='foreground_below_original32_pixels',GT_read=False)
    return full,dict(status='valid',box=out,initial_labels=initial,final_labels=mask,entire_foreground_retained=True,source_support_box=[x0/w,y0/h,x1/w,y1/h],GT_read=False)

def contracts():
    sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
    import cv2,numpy as np
    cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False);rgb=np.full((100,120,3),230,np.uint8);rgb[20:80,35:85]=[20,40,100]
    m,d=foreground(rgb,[.25,.15,.75,.85],100);assert m is not None and d['entire_foreground_retained'];assert m[45,50]and not m[0,0]
    m2,d2=foreground(rgb,[.25,.15,.75,.85],100);assert np.array_equal(m,m2)
    for box in ([1.1,.2,1.2,.5],[.2,.8,.1,.9]):assert foreground(rgb,box,100)[0]is None
    assert np.all(np.isin(d['initial_labels'],[0,1,2,3]))
    return dict(status='pass',count=5,scope='synthetic pixel-only, full foreground, deterministic local seed, invalid box fallback; no GT/model',CPU_only=True,GT_read=False)

def prepare():
    sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
    import cv2,numpy as np
    from PIL import Image
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import stvg_evidence_spatial_inputs009 as wrapper
    from scripts import stvg_weightfree_spatial_v1 as spatial
    from scripts.run_stvg_weightfree_expand_v1 import selected_rgb
    from scripts.prepare_stvg_short_window_teacher_v1 import render_context_frames
    builder=wrapper._base;verify();sys.addaudithook(guard);cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    assert read(BASE/'temporal/ROOT_READBACK.json')['status']=='pass';put(SP/'CPU_CONTRACTS.json',contracts());jobs=[];files={};counts=[]
    times={r['case_id']:r['selected_interval']for r in read(BASE/'temporal/TEMPORAL_LOCK.json')['records']}
    def save(p,raw):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert p.read_bytes()==raw
        else:p.write_bytes(raw)
        files[str(p.relative_to(ROOT))]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=rec['probe_alias'];case,cp=load_case(rec);req=read(BASE/'temporal/requests'/f'{cid}.json');rp=BASE/'temporal/requests'/f'{cid}.json';interval=times[cid]
        try:centers=builder.nearest_unique_grid_centers(interval,case['frame_ids'],[r['physical_frame_id']for r in req['frame_map']])
        except ValueError as e:
            counts.append(dict(case_id=cid,status='unknown_insufficient_centers',reason=str(e),GT_read=False));continue
        needed={int(w['center_physical_frame_id'])+k for w in centers for k in range(-2,3)};rgb=selected_rgb(req,needed)
        oldpath=LITE/'reclocked_inputs/new_crop/proposals'/f'{cid}.json';old=read(oldpath);cache={};mcache={}
        for w in old['windows']:
            fid=w['center_physical_frame_id'];cache[fid]=w;mp=ROOT/w['mask_archive_path'];assert sha(mp)==w['mask_archive_sha256']
            with np.load(mp,allow_pickle=False)as a:mcache[fid]=np.asarray(a['masks']).copy()
        pool=builder.build_candidate_pool(case,req,interval,cid,rgb,source_case_sha256=case['source_case_sha256'],request_sha256=sha(rp),cached_windows_by_center=cache,cached_masks_by_center=mcache)
        windows=pool['windows'];contextids=[w['center_physical_frame_id']for w in windows];context_times=[next(r['source_timestamp_s']for r in req['frame_map']if r['physical_frame_id']==fid)for fid in contextids]
        ctx,_=render_context_frames([Image.fromarray(rgb[f])for f in contextids]);ctx=np.ascontiguousarray(ctx.transpose(0,3,1,2));ctxraw=builder._npy_bytes(ctx)
        updated=copy.deepcopy(windows);newcount=0
        for w in updated:
            fid=w['center_physical_frame_id'];frames=[rgb[f]for f in w['physical_frame_ids']];small=[builder._resize_small(a,320)for a in frames];mid=w['physical_frame_ids'].index(fid)
            seed=int(hashlib.sha256((rec['query_key_sha256']+':foreground:'+str(fid)).encode()).hexdigest()[:8],16)
            tick=time.perf_counter();mask,info=foreground(small[mid],w['source_anchor_raw_box_xyxy'],seed);info['generation_seconds']=time.perf_counter()-tick
            ap=SP/'private_intermediates'/cid/f'{w["slot"]}.npz'
            if mask is not None:
                ap.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ap,initial_labels=info.pop('initial_labels'),final_labels=info.pop('final_labels'),full_foreground=mask.astype(np.uint8));files[str(ap.relative_to(ROOT))]=dict(path=str(ap.relative_to(ROOT)),sha256=sha(ap),bytes=ap.stat().st_size)
                gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)for x in small];flows=[]
                for a,b in zip(gray,gray[1:]):
                    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);flows.append((dis.calc(a,b,None),dis.calc(b,a,None)))
                mm=[np.zeros_like(mask)for _ in small];mm[mid]=mask
                for k in range(mid,len(mm)-1):mm[k+1]=spatial.warp(mm[k],*flows[k])
                for k in range(mid-1,-1,-1):mm[k]=spatial.warp(mm[k+1],flows[k][1],flows[k][0])
                boxes=[spatial.bbox(x)for x in mm];ident=min(9,len(w['tracks'])+1)
                track=dict(id=ident,candidate_id=ident,role='appearance_region',boxes=boxes,center_box=boxes[mid],presence=sum(x is not None for x in boxes)/len(boxes),proposal_rank=-1,score=0.,mask_bbox_source='prospective_Source_seeded_full_foreground',GT_read=False)
                keep=w['tracks'][:ident-1];keepm=w['_masks'][:ident-2];w['tracks']=keep+[track];w['_masks']=np.concatenate([keepm,np.stack(mm)[None].astype(np.uint8)],axis=0);newcount+=1
            w['new_hypothesis_status']=info
        for group,ww in [('E0',windows),('E1',updated)]:
            masks={}
            for w in ww:
                w.update(interval_half_open=copy.deepcopy(interval),source_case_sha256=case['source_case_sha256'],request_sha256=sha(rp),context_center_physical_frame_ids=contextids,context_timestamps_seconds=context_times)
                masks[w['slot']]=np.asarray(w.pop('_masks'),dtype=np.uint8);mp=SP/group/'masks'/f'{cid}_{w["slot"]}.npz';save(mp,builder._npz_bytes(masks=masks[w['slot']]))
                w.update(mask_archive_path=str(mp.relative_to(ROOT)),mask_archive_sha256=sha(mp))
            pp=SP/group/'proposals'/f'{cid}.json';put(pp,dict(status='complete',case_id=cid,dataset=rec['dataset'],windows=ww,source_case_sha256=case['source_case_sha256'],GT_read=False));files[str(pp.relative_to(ROOT))]=dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp),bytes=pp.stat().st_size)
            jj,ff=builder._build_jobs_for_condition(condition='new_crop',condition_base=SP/group,case_id=cid,dataset=rec['dataset'],window_rows=ww,context_chw=ctx,context_raw=ctxraw,rgb_by_physical_id=rgb,query_text=req['query_text'],masks_by_slot=masks)
            for rel,raw in ff.items():save(ROOT/rel,raw)
            for j in jj:
                j['job_id']=group+'_'+j['job_id'];j['evidence_group']=group;j['time_encoding_version']='fractional_uniform_v2';jobs.append(j)
        counts.append(dict(case_id=cid,dataset=rec['dataset'],status='ready',new_hypotheses=newcount,reused_centers=pool['reused_cached_center_physical_ids'],fresh_centers=pool['fresh_center_physical_ids'],GT_read=False))
        write(SP/'STATUS.json',dict(status='running',stage='candidate_inputs',done=len(counts),total=32,GT_read=False),mutable=True);print('SPATIAL_INPUT',len(counts),32,cid,newcount,flush=True);del case,rgb,pool,windows,updated;gc.collect()
    put(SP/'TEACHER_JOB_LOCK.json',dict(status='locked',records=jobs,count=len(jobs),GT_read=False));put(SP/'CANDIDATE_INPUT_BARRIER.json',dict(status='complete',count=32,teacher_job_count=len(jobs),files=list(files.values()),case_records=counts,GT_read=False))
    rt=copy.deepcopy(read(LITE/'reclocked_inputs/TEACHER_RUNTIME.json'))
    for p in [Path(__file__),ROOT/'scripts/stvg_instance_temporal32_v1.py',BASE/'EVIDENCE_MODIFICATION_LOCK.json',BASE/'temporal/TEMPORAL_LOCK.json',SP/'TEACHER_JOB_LOCK.json',SP/'CANDIDATE_INPUT_BARRIER.json']:
        rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    rt.update(status='locked',GT_read=False,scope='common corrected Temporal/Spatial interface; only E1 last proposal replacement; no GT or extra prompt')
    put(SP/'TEACHER_RUNTIME.json',rt)

def input_root():
    import numpy as np
    bar=read(SP/'CANDIDATE_INPUT_BARRIER.json');assert bar['count']==32 and bar['GT_read']is False
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    for j in read(SP/'TEACHER_JOB_LOCK.json')['records']:
        assert hashlib.sha256(j['prompt'].encode()).hexdigest()==j['prompt_sha256'];assert j['allowed_output_ids']==list(range(j['candidate_count']+1))
        for v in j['videos']:
            a=np.load(ROOT/v['path'],allow_pickle=False);assert hashlib.sha256(a.tobytes()).hexdigest()==v['tensor_sha256']and sha(ROOT/v['path'])==v['sha256']
        e0=read(SP/'E0/proposals'/f"{j['case_id']}.json")['windows'][j['center_slot']];e1=read(SP/'E1/proposals'/f"{j['case_id']}.json")['windows'][j['center_slot']]
        assert e0['physical_frame_ids']==e1['physical_frame_ids']and e0['context_center_physical_frame_ids']==e1['context_center_physical_frame_ids']
        if e1['new_hypothesis_status']['status']=='valid':assert e0['tracks'][:8]==e1['tracks'][:8]if len(e0['tracks'])==9 else e0['tracks']==e1['tracks'][:-1]
        else:assert e0['tracks']==e1['tracks']
    put(SP/'ROOT_INPUT_READBACK.json',dict(status='pass',count=32,job_count=bar['teacher_job_count'],GT_read=False,all_NPY_mask_proposal_job_hashes_verified=True,common_clocks_context_preserved=True,barrier_sha256=sha(SP/'CANDIDATE_INPUT_BARRIER.json')))
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='contracts':print(json.dumps(contracts()))
    else:{'prepare':prepare,'root':input_root}[mode]()
