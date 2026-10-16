"""Prelock32 parents and preserve only global RGB while current media is live."""
from pathlib import Path
import copy,gc,hashlib,io,json,os,select,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE as CORE,read,put,sha,guard
BASE=ROOT/'artifacts/stvg_global32_context_v1'
def verify():
    for p,h in read(BASE/'PREPARATION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def setup():
    BASE.mkdir(exist_ok=False)
    rows=read(CORE/'ROSTER_LOCK.json')['records'];chosen=[]
    for ds in ('hc2','vidstg'):chosen+=sorted([r for r in rows if r['dataset']==ds],key=lambda r:(r['parent_sha256'],r['query_key_sha256']))[:16]
    assert len(chosen)==len({(r['dataset'],r['parent_sha256']) for r in chosen})==32
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=32,per_domain=16,records=chosen,selection='first16 parent hashes/domain from prelocked256; before new WHO, GT or outcome reading',GT_read=False))
    cfg=read(ROOT/'checkpoints/STVG-R1-7B/config.json')
    assert cfg['max_position_embeddings']==128000 and cfg['sliding_window']==32768
    assert [cfg['vision_config'][k] for k in ('patch_size','temporal_patch_size','spatial_merge_size')]==[14,2,2]
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=32,per_domain=16,global_frames=32,global_tokens_before=4096,global_tokens_kept=512,local_frames=5,local_tokens_kept=96,WHO_calls_per_query=4,WHO_generation_cap=96,context_scope='complete legal input frame_map, not GT or QA interval',frame_selection='nearest integer np.linspace(0,N-1,32) including both legal endpoints',short_input_policy='if fewer than32 distinct input frames, retain in cohort as unavailable and use fixed Temporal fallback; do not substitute favorable sources',clock='physical source timestamp list plus original fractional mean-gap uniform encoder clock, not exact irregular timestamps',control='exact same-source A from core256; no new control GPU or scoring',local_indices='exact core A native feature bytes and local96 indices; no candidate geometry/token-selection changes',global_selection='same original A algorithm/constants at12.5%, with T16; pure-A CPU identity contracts',global_encoder='once per distinct qualified global pixel/model/processor/clock binding; four distinct LM prefills still counted; qualification extra encodes reported separately',qualification='each domain first eligible input: global-only versus joint encoding exact; core local versus joint exact; raw no-prune original-generate first8 exact; compressed KV/recompute next2 exact',Student='same original A configuration and episodic1792 Gaussian32 fullstep OPD, original admission/rejection/rollback; full model re-decode',semantic='all32 parents slots0/3 arm-blind candidate identity/role/full-extent labels, missing unknown; no GT selection',readouts=['identity_role_extent','selected_reference_geometry','accepted_nonzero_updates','full_Student_metrics','encoder_prefill_generation_Student_cost'],claims='joint global range/density/budget intervention; not isolated compression-algorithm gain, not assumed faster or correct',ordered_prompt='deferred',joint_four_windows=False,direct_preference_OPD=False,temporal_change=False,bootstrap=False,CURRENT_change=False,official_fullset=False,GT_read=False))
    files=[Path(__file__),ROOT/'scripts/stvg_global32_native_v1.py',ROOT/'scripts/stvg_referent_abc_native002.py',ROOT/'scripts/stvg_referent_abc_tokens002.py',ROOT/'scripts/prepare_stvg_short_window_teacher_v1.py',ROOT/'scripts/run_stvg_weightfree_expand_v1.py',CORE/'ROSTER_LOCK.json',CORE/'EXECUTION_SCOPE_LOCK003.json',CORE/'NATIVE_DISPATCH_RUNTIME003.json',CORE/'temporal/ROOT_READBACK.json',ROOT/'checkpoints/STVG-R1-7B/config.json',BASE/'ROSTER_LOCK.json',BASE/'DESIGN_LOCK.json']
    files += [CORE/'temporal/requests'/(r['probe_alias']+'.json') for r in chosen]
    put(BASE/'PREPARATION_RUNTIME.json',dict(status='locked',count=32,pins={str(p.relative_to(ROOT)):sha(p) for p in files},CPU_only=True,GT_read=False))
    put(BASE/'STATUS.json',dict(status='prepared',stage='global_RGB_only_then_core6_wait',count=32,GT_read=False))
def prepare_rgb():
    sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
    import cv2,numpy as np
    from PIL import Image
    from scripts.prepare_stvg_short_window_teacher_v1 import _letterbox
    from scripts.run_stvg_weightfree_expand_v1 import selected_rgb
    from scripts.stvg_native_instance_common_v1 import write
    verify();sys.addaudithook(guard);cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False);files=[]
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=rec['probe_alias'];rp=CORE/'temporal/requests'/(cid+'.json');req=read(rp)
        rows=sorted(req['frame_map'],key=lambda r:int(r['local_frame_index']));ids=[int(r['physical_frame_id']) for r in rows]
        assert len(set(ids))==len(ids) and all(b>a for a,b in zip(ids,ids[1:]))
        if len(ids)<32:row=dict(case_id=cid,dataset=rec['dataset'],status='unavailable_less_than32_distinct_input_frames',GT_read=False)
        else:
            ix=np.rint(np.linspace(0,len(rows)-1,32)).astype(int);assert len(set(ix.tolist()))==32 and ix[0]==0 and ix[-1]==len(rows)-1
            chosen=[rows[int(i)] for i in ix];fids=[int(r['physical_frame_id']) for r in chosen];times=[float(r['source_timestamp_s']) for r in chosen]
            assert all(b>a for a,b in zip(times,times[1:]))
            assert sha(Path(req['video_path']))==req['canonical_video_sha256'];rgb=selected_rgb(req,set(fids));arrays=[];render=[]
            for fid in fids:
                im,meta=_letterbox(Image.fromarray(rgb[fid]));a=np.ascontiguousarray(np.asarray(im,dtype=np.uint8));arrays.append(a)
                render.append(dict(letterbox=meta,source_rgb_sha256=hashlib.sha256(rgb[fid].tobytes()).hexdigest(),rendered_rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_overlays=[]))
            a=np.ascontiguousarray(np.stack(arrays).transpose(0,3,1,2));p=BASE/'global_inputs'/(cid+'.npy');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.save(p,a,allow_pickle=False)
            row=dict(case_id=cid,dataset=rec['dataset'],status='ready',path=str(p.relative_to(ROOT)),sha256=sha(p),tensor_sha256=hashlib.sha256(a.tobytes()).hexdigest(),bytes=p.stat().st_size,physical_frame_ids=fids,timestamps_seconds=times,renderer=render,request_sha256=sha(rp),canonical_sha256=req['canonical_video_sha256'],GT_read=False)
            del rgb,a,arrays;gc.collect()
        files.append(row);write(BASE/'RGB_STATUS.json',dict(status='running',stage='global_RGB_CPU_only',done=len(files),total=32,GT_read=False),mutable=True)
    put(BASE/'GLOBAL_RGB_BARRIER.json',dict(status='complete',count=32,records=files,CPU_only=True,GT_read=False))
    root_rgb()
def root_rgb():
    import numpy as np
    verify();rows=read(BASE/'GLOBAL_RGB_BARRIER.json')['records'];assert len(rows)==32
    for r in rows:
        if r['status']!='ready':continue
        p=ROOT/r['path'];a=np.load(p,allow_pickle=False);assert sha(p)==r['sha256'] and a.shape==(32,3,448,448) and a.dtype==np.uint8 and hashlib.sha256(a.tobytes()).hexdigest()==r['tensor_sha256']
        req=read(CORE/'temporal/requests'/(r['case_id']+'.json'));fm=sorted(req['frame_map'],key=lambda z:int(z['local_frame_index']));ix=np.rint(np.linspace(0,len(fm)-1,32)).astype(int)
        assert r['physical_frame_ids']==[int(fm[int(i)]['physical_frame_id']) for i in ix] and r['timestamps_seconds']==[float(fm[int(i)]['source_timestamp_s']) for i in ix]
    put(BASE/'ROOT_RGB_READBACK.json',dict(status='pass',count=32,CPU_only=True,GT_read=False,barrier_sha256=sha(BASE/'GLOBAL_RGB_BARRIER.json'),full_legal_endpoints_and_all_tensor_hashes=True))
def jobs():
    from scripts.stvg_evidence_spatial_inputs005 import _confirmation003_prompt,_video_record
    verify();assert read(BASE/'ROOT_RGB_READBACK.json')['status']=='pass';assert read(CORE/'SPATIAL_INPUT_HANDOFF.json')['count']==256
    import numpy as np
    rgb={r['case_id']:r for r in read(BASE/'GLOBAL_RGB_BARRIER.json')['records']};out=[]
    for orig in read(CORE/'TEACHER_JOB_LOCK.json')['records']:
        r=rgb.get(orig['case_id'])
        if r is None or r['status']!='ready':continue
        j=copy.deepcopy(orig);req=read(CORE/'temporal/requests'/(j['case_id']+'.json'));times=r['timestamps_seconds'];ids=[x['id'] for x in j['candidate_bindings']]
        oldtimes=orig['videos'][0]['actual_timestamps_seconds']
        assert orig['prompt']==_confirmation003_prompt(req['query_text'],oldtimes,orig['videos'][1]['actual_timestamps_seconds'],ids)
        p=ROOT/r['path'];a=np.load(p,allow_pickle=False);j['videos'][0]=_video_record(r['path'],p.read_bytes(),a,times,'preserved_unmarked_full_input_context32')
        j['prompt']=_confirmation003_prompt(req['query_text'],times,orig['videos'][1]['actual_timestamps_seconds'],ids);j['prompt_sha256']=hashlib.sha256(j['prompt'].encode()).hexdigest();j['context_center_physical_frame_ids']=r['physical_frame_ids'];j['global_context_design']='full_input32_keep512';j['original_job_sha256']=hashlib.sha256(json.dumps(orig,sort_keys=True).encode()).hexdigest()
        assert j['videos'][1]==orig['videos'][1] and j['candidate_bindings']==orig['candidate_bindings'] and j['center_physical_frame_id']==orig['center_physical_frame_id'];out.append(j)
    put(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',count=32,job_count=len(out),records=out,GT_read=False))
    put(BASE/'ROOT_INPUT_READBACK.json',dict(status='pass',count=32,job_count=len(out),same_local_NPY_geometry_IDs_times_and_original_prompt_wording=True,GT_read=False,job_lock_sha256=sha(BASE/'TEACHER_JOB_LOCK.json'),core_input_root_sha256=sha(CORE/'spatial/ROOT_INPUT_READBACK.json')))
if __name__=='__main__':
    {'setup':setup,'rgb':prepare_rgb,'jobs':jobs}[sys.argv[1]]()
