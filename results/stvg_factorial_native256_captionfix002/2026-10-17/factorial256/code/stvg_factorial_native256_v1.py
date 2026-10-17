"""True Native-clock A/B spatial evidence; original algorithms and cohort256."""
from pathlib import Path
import ast,collections,copy,gc,hashlib,inspect,json,os,subprocess,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as old
BASE=ROOT/'artifacts/stvg_factorial_native256_v1';CORE=old.CORE
read,sha,put=old.read,old.sha,old.write
CURRENT_CASE=None
def verify():
    old.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME001.json')['pins'].items():assert sha(ROOT/p)==h,p
def setup():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    BASE.mkdir(exist_ok=False);rr=read(CORE/'ROSTER_LOCK.json');put(BASE/'ROSTER_LOCK.json',rr);put(BASE/'SOURCE_CAPTURE_BARRIER.json',read(CORE/'SOURCE_CAPTURE_BARRIER.json'));put(BASE/'EVIDENCE_MODIFICATION_LOCK.json',read(CORE/'EVIDENCE_MODIFICATION_LOCK.json'))
    rows=[];caps=[];identical=[];selected={r['case_id']:r for r in read(CORE/'TEMPORAL_LOCK.json')['records']};elig={r['case_id']:r for r in read(CORE/'temporal/INPUT_BARRIER.json')['records']};(BASE/'temporal/requests').mkdir(parents=True)
    for rec in rr['records']:
        cid=rec['probe_alias'];native=elig[cid]['legacy_interval'];rows.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=selected[cid]['source_case_sha256'],selected_interval=native,GT_read=False));p=CORE/'temporal/requests'/(cid+'.json');(BASE/'temporal/requests'/(cid+'.json')).write_bytes(p.read_bytes());cp=CORE/'temporal/teacher'/cid/'time_Native.json';available=cp.exists() and bool(read(cp)['raw_caption'].strip());caps.append(dict(case_id=cid,candidate_id='time_Native',path=str(cp.relative_to(ROOT)) if available else None,sha256=sha(cp) if available else None,available=available,GT_read=False))
        if native==selected[cid]['selected_interval']:identical.append(cid)
    put(BASE/'temporal/TEMPORAL_LOCK.json',dict(status='locked',count=256,records=rows,clock='Source Native, zero new Temporal parameter updates',GT_read=False));put(BASE/'TEMPORAL_LOCK.json',read(BASE/'temporal/TEMPORAL_LOCK.json'));put(BASE/'temporal/ROOT_READBACK.json',dict(status='pass',count=256,Source_Native_intervals_exact=True,new_Temporal_parameter_updates=0,GT_read=False));put(BASE/'CAPTION_LOCK.json',dict(status='locked',count=256,records=caps,selection='only already-sealed time_Native candidate caption; no adapted interval/caption selection or Temporal fitting',GT_read=False));put(BASE/'SOURCE_GEOMETRY.json',read(CORE/'SOURCE_GEOMETRY.json'))
    put(BASE/'STUDY_LOCK.json',dict(status='locked',count=256,per_domain=128,development_only=True,confirmation128_not_reopened=True,arms=['Frozen','Temporal','S_A_Native','S_B_Native','Full_A_Adapted','Full_B_Adapted','T_S_A_Native_evidence','T_S_B_Native_evidence'],new_model_groups=['Native-clock A','Native-clock B'],same_Source_checkpoints_candidate_algorithm_Prompt_seed_rules_parameters_rejection=True,Native_caption_reused_without_Temporal_update=True,spatial_fit_does_not_depend_on_final_interval=True,final_Temporal_metadata_only_from_existing_T=True,Native_interval_unchanged_sources=identical,matched_final_time_evidence_contrast=True,interactions='(Full-T)-(SNative-F); includes observation/reference changes, not pure gradient synergy',fixed_time_geometry='all known GT frames and same exact temporal output, not only vIoU interaction',metrics=['vIoU','tIoU','sIoU_dense_GT','R@0.3','R@0.5'],uncertainty='paired source-level 95% Student-t intervals, separately by domain; no significance/selection gate and no bootstrap',CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),GT_read=False))
    for a in ('A','B'):
        b=BASE/'arms'/a;b.mkdir(parents=True);put(b/'ROSTER_LOCK.json',rr);put(b/'SCOPE_CONFIG.json',dict(label='true Native-clock Spatial-only '+a,core_scope=str(BASE.relative_to(ROOT)),design_template=str((CORE/'trials'/a/'DESIGN_LOCK.json').relative_to(ROOT)),policy=a,recipe='lite_fullstep',global_frames=4,global_keep=64,local_keep=96,physical_clock=False,vision_reuse_scope=None,ablation=None,local_control=a,count=256,GT_read=False))
    files=[Path(__file__),ROOT/'scripts/continue_stvg_factorial_native256_v1.py',ROOT/'scripts/score_stvg_factorial_native256_v1.py',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'temporal/TEMPORAL_LOCK.json',BASE/'CAPTION_LOCK.json',BASE/'STUDY_LOCK.json',ROOT/'scripts/stvg_instance_spatial32_v1.py',ROOT/'scripts/stvg_overnight_global_v1.py',ROOT/'scripts/stvg_overnight_student_v1.py',ROOT/'scripts/run_stvg_spatial_controls256_native_v1.py']
    put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in files},deadline_epoch=time.time()+28800,scope='new separately authorized finite factorial, original night protocol preserved closed',GT_read=False));put(BASE/'STATUS.json',dict(status='prepared',stage='Native_clock_E1_CPU_inputs',count=256,GT_read=False));print('NATIVE_FACTORIAL256_LOCKED')
def native_rgb(request,wanted):
    """Exact original FFmpeg RGB stream, hash-matched to retired FFV1 pixels."""
    import numpy as np
    from scripts import run_stvg_table1_native_instance_adapter_v1 as native
    case=CURRENT_CASE;identity=native._source_identity(case);assert identity['source_video_sha256']==request['source_video_sha256'];source=Path(identity['video_path']);assert sha(source)==identity['source_video_sha256'];ffmpeg,_=native._ffmpeg_identity();cmd=[str(ffmpeg),'-v','error','-threads','2','-i',str(source)];
    if identity['dataset']=='vidstg':cmd+=['-vsync','0']
    cmd+=['-f','rawvideo','-pix_fmt','rgb24','pipe:1'];size=identity['width']*identity['height']*3;digest=hashlib.sha256();count=0;out={};wanted=set(wanted);log=BASE/'pixel_readback'/(case['case_id']+'.stderr');log.parent.mkdir(exist_ok=True)
    with log.open('xb') as err:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err,bufsize=0)
        try:
            while True:
                data=bytearray()
                while len(data)<size:
                    z=p.stdout.read(size-len(data))
                    if not z:break
                    data.extend(z)
                if not data:break
                assert len(data)==size
                if identity['start']<=count<identity['declared_end']:
                    digest.update(data)
                    if count in wanted:out[count]=np.frombuffer(data,dtype=np.uint8).reshape(identity['height'],identity['width'],3).copy()
                count+=1
            p.stdout.close();assert p.wait()==0
        finally:
            if p.poll() is None:p.terminate();p.wait()
    assert digest.hexdigest()==request['canonical_pixel_sha256'] and set(out)==wanted
    put(BASE/'pixel_readback'/(case['case_id']+'.json'),dict(status='pass',Source_RGB_stream_equals_original_canonical_pixel_SHA=True,source_sha256=identity['source_video_sha256'],canonical_pixel_sha256=digest.hexdigest(),selected_frame_count=len(out),no_canonical_video_reencoding=True,GT_read=False));return out
def prepare():
    from scripts import stvg_instance_spatial32_v1 as original
    from scripts.stvg_native_instance_common_v1 import guard
    src=inspect.getsource(original.prepare);src=src.replace('from scripts.run_stvg_weightfree_expand_v1 import selected_rgb','from scripts.run_stvg_weightfree_expand_v1 import selected_rgb as unused_old_decoder')
    start=src.index('oldpath=LITE/');end=src.index('pool=builder.build_candidate_pool',start);src=src[:start]+'cache={};mcache={}\n        '+src[end:];src=src.replace("[('E0',windows),('E1',updated)]","[('E1',updated)]")
    needle="cid=rec['probe_alias'];case,cp=load_case(rec);";assert src.count(needle)==1;src=src.replace(needle,needle+'set_current(case);')
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(256),n) if type(n.value)is int and n.value==32 else n
    def current(case):
        global CURRENT_CASE
        CURRENT_CASE=case
    ns=dict(original.__dict__);ns.update(BASE=BASE,SP=BASE/'spatial',verify=verify,selected_rgb=native_rgb,set_current=current);exec(compile(ast.fix_missing_locations(Count().visit(ast.parse(src))),__file__,'exec'),ns);ns['prepare']()
def input_root():
    import numpy as np
    verify();bar=read(BASE/'spatial/CANDIDATE_INPUT_BARRIER.json');assert bar['count']==256
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    jobs=read(BASE/'spatial/TEACHER_JOB_LOCK.json')['records'];times={r['case_id']:r['selected_interval'] for r in read(BASE/'temporal/TEMPORAL_LOCK.json')['records']}
    for j in jobs:
        pool=read(BASE/'spatial/E1/proposals'/(j['case_id']+'.json'));w=next(w for w in pool['windows'] if w['slot']==j['center_slot']);assert w['interval_half_open']==times[j['case_id']] and j['center_physical_frame_id']==w['center_physical_frame_id']
        for v in j['videos']:
            p=ROOT/v['path'];a=np.load(p,allow_pickle=False);assert sha(p)==v['sha256'] and hashlib.sha256(a.tobytes()).hexdigest()==v['tensor_sha256']
    put(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',count=256,job_count=len(jobs),records=jobs,GT_read=False));put(BASE/'SPATIAL_INPUT_HANDOFF.json',dict(status='complete_pending_native_evidence',count=256,job_count=len(jobs),GT_read=False));put(BASE/'ROOT_INPUT_READBACK.json',dict(status='pass',count=256,all_Native_clocks_and_pixels_exact=True,job_count=len(jobs),GT_read=False))
def features():
    import torch,numpy as np
    from scripts import run_stvg_spatial_controls256_native_v1 as g
    from scripts.stvg_native_instance_common_v1 import guard,write
    verify();sys.addaudithook(guard);g.BASE=BASE;m,p,device,proof,secs=g.load();files=[];reused=0;retired=[];originals={(j['case_id'],j['center_slot']):j for j in read(CORE/'TEACHER_JOB_LOCK.json')['records']}
    for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']:
        oldj=originals[j['case_id'],j['center_slot']];oldpath=CORE/'native_cache'/oldj['job_id']/'native.npz';frozen=None
        same=j['prompt_sha256']==oldj['prompt_sha256'] and all(all(a[k]==b[k] for k in ('tensor_sha256','fps')) for a,b in zip(j['videos'],oldj['videos']))
        prior=read(oldpath.with_name('POLICIES.json'));assert sha(oldpath)==prior['native_sha256']
        if same:
            with np.load(oldpath,allow_pickle=False) as z:frozen=z['features'].copy()
            reused+=1
        b,ps,e,pos,f,runs,pol,v,ma,gr,q,caption,available=g.prepare(m,p,device,j,frozen=frozen);g.cache_write(j,proof,ps,f,pol,v,ma,gr,q,caption,available);path=BASE/'native_cache'/j['job_id']/'POLICIES.json';files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),case_id=j['case_id']));del b,e,pos,f,runs,pol,frozen;gc.collect();torch.cuda.empty_cache()
        # Old core R1 features have no remaining consumer after the coverage
        # group seal and this exact-input cache decision. Retain policies,
        # responses, predictions, Source prefixes and all original NPY pixels.
        entry=dict(path=str(oldpath.relative_to(ROOT)),sha256=prior['native_sha256'],bytes=oldpath.stat().st_size,used_before_retirement=True,GT_read=False);put(BASE/'retired_core_feature_receipts'/(j['job_id']+'.json'),entry);oldpath.unlink();retired.append(entry)
        write(BASE/'FEATURE_STATUS.json',dict(status='running',done=len(files),total=len(read(BASE/'TEACHER_JOB_LOCK.json')['records']),count=256,GT_read=False),mutable=True)
    put(BASE/'FEATURE_BARRIER.json',dict(status='complete',count=256,new_encoder_calls=len(files)-reused,exact_old_visual_encoder_reuses=reused,files=files,model_load_seconds=secs,GT_read=False));put(BASE/'CORE_CACHE_RETIREMENT.json',dict(status='complete',count=len(retired),bytes=sum(x['bytes'] for x in retired),only_old_R1_native_npz=True,original_NPY_Source_Teacher_predictions_preserved=True,GT_read=False))
def configure(module,arm):
    verify();scope=BASE/'arms'/arm;module.BASE=scope;module.CORE=BASE;module.N=256;module.CFG=read(scope/'SCOPE_CONFIG.json');module.common=types.SimpleNamespace(BASE=BASE,CORE=CORE,verify=verify,deadline=lambda:read(BASE/'EXECUTION_RUNTIME001.json')['deadline_epoch'],read=read,write=put,sha=sha);return module
def arm_stage(arm,stage):
    if stage in ('jobs','qualification','teacher','teacher_root'):
        from scripts import stvg_overnight_global_v1 as g
        configure(g,arm);return {'jobs':g.jobs,'qualification':g.qualify,'teacher':g.teacher,'teacher_root':g.teacher_root}[stage]()
    from scripts import stvg_overnight_student_v1 as s
    configure(s,arm)
    if stage=='student_prepare':
        # Fit itself never reads selected_interval; original _build_arms attaches
        # Native and existing Adapted intervals to the SAME fully read trajectory.
        src=inspect.getsource(s.prepare);needle="read(CORE/'TEMPORAL_LOCK.json')";assert src.count(needle)==1;src=src.replace(needle,"read(ORIGINAL_CORE/'TEMPORAL_LOCK.json')");ns=dict(s.__dict__);ns['ORIGINAL_CORE']=CORE;exec(compile(src,__file__,'exec'),ns);return ns['prepare']()
    return {'student_worker':s.worker,'student_root':s.root}[stage]()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('setup','prepare','input_root','features'):globals()[mode]()
    else:arm_stage(sys.argv[2],mode)
