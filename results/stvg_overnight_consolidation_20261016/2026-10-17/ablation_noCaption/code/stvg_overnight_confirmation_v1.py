"""Locked128 new-parent confirmation using unchanged source/temporal/E1 interfaces."""
from pathlib import Path
import ast,copy,gc,hashlib,inspect,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
read,sha,put=c.read,c.sha,c.write
BASE=c.BASE/'confirmation128';N=128
def verify():
    c.verify()
    for p,h in read(c.BASE/'CONFIRMATION_DISPATCH_RUNTIME002.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert read(BASE/'ROSTER_LOCK.json')['count']==128
def transform(fn,old,new,overrides):
    src=inspect.getsource(fn)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(new),n) if type(n.value)is int and n.value==old else n
    tree=Count().visit(ast.parse(src));ns=dict(sys.modules[fn.__module__].__dict__);ns.update(overrides);exec(compile(ast.fix_missing_locations(tree),str(Path(__file__)),'exec'),ns);return ns[fn.__name__]
def setup():
    verify();rows=read(BASE/'ROSTER_LOCK.json')['records'];core=read(c.CORE/'ROSTER_LOCK.json')['records'];assert not {(x['dataset'],x['parent_sha256']) for x in rows}&{(x['dataset'],x['parent_sha256']) for x in core}
    rt=read(c.BASE/'CONFIRMATION_DISPATCH_RUNTIME002.json');original=read(ROOT/'artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json');put(BASE/'SOURCE_RUNTIME.json',dict(status='locked',count=N,pins=rt['pins'],source_checkpoints=original['source_checkpoints'],original_Source_runtime_sha256=sha(ROOT/'artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json'),GT_read=False))
    # Membership was fixed before this cache inventory. No broad payload search.
    known={}
    for scope in (c.CORE,c.OLDGLOBAL,ROOT/'artifacts/stvg_soft_ep_coordinate128_v1',ROOT/'artifacts/stvg_lite_event_support_v1/confirmation'):
        rp=scope/'ROSTER_LOCK.json';bp=scope/'SOURCE_CAPTURE_BARRIER.json'
        if not rp.exists() or not bp.exists():continue
        files={r['case_id']:r for r in read(bp)['files']}
        for r in read(rp)['records']:
            if r['probe_alias'] in files:known.setdefault((r['dataset'],r['query_key_sha256']),dict(record=r,file=files[r['probe_alias']],receipt_sha256=sha((ROOT/files[r['probe_alias']]['path']).with_suffix('.receipt.json'))))
    entries=[]
    for r in rows:
        x=known.get((r['dataset'],r['query_key_sha256']))
        if x:
            for k in ('parent_sha256','plan_sha256','subject_sha256','video_sha256','frame_ids','width','height','fps','video_bounds'):assert x['record'][k]==r[k],k
        entries.append(dict(case_id=r['probe_alias'],dataset=r['dataset'],cached=x,GT_read=False))
    put(BASE/'SOURCE_CACHE_INVENTORY.json',dict(status='locked',count=N,records=entries,selection_already_fixed=True,GT_read=False))
    p=c.CORE/'EVIDENCE_MODIFICATION_LOCK.json';d=read(p);d['count']=N;d['original_core_lock_sha256']=sha(p);put(BASE/'EVIDENCE_MODIFICATION_LOCK.json',d)
    for name in ('TEMPORAL_DISPATCH_RUNTIME.json','INPUT_DISPATCH_RUNTIME.json'):put(BASE/name,dict(status='locked',count=N,pins=rt['pins'],GT_read=False))
    c.reserve('confirmation_temporal128',Temporal=512)
def source():
    import numpy as np
    from scripts import prepare_stvg_spatial_controls256_v1 as old
    src=inspect.getsource(old.source);assert src.count("c['frame_ids']==rec['frame_ids']")==1;src=src.replace("c['frame_ids']==rec['frame_ids']","np.array_equal(c['frame_ids'],rec['frame_ids'])")
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(128),n) if type(n.value)is int and n.value==256 else n
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify);exec(compile(ast.fix_missing_locations(Count().visit(ast.parse(src))),str(Path(__file__)),'exec'),ns);ns['source']()
def temporal(mode):
    from scripts import run_stvg_spatial_controls256_temporal_v1 as original,stvg_instance_temporal32_v1 as old
    out=BASE/'temporal'
    def check():
        verify()
        for p,h in read(out/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if mode=='prepare':transform(original.prepare,256,N,dict(BASE=BASE,OUT=out,verify=verify))()
    else:
        fn=old.teacher if mode=='teacher' else old.fit_cases
        f=transform(fn,32,N,dict(BASE=BASE,OUT=out,verify_temporal=check,p0_reuse=lambda *args:None))
        f(True) if mode=='readback' else f()
def spatial(mode):
    from scripts import run_stvg_spatial_controls256_inputs_v1 as old
    def check():verify();assert read(BASE/'temporal/ROOT_READBACK.json')['count']==N
    fn=old.prepare if mode=='prepare' else old.input_root
    transform(fn,256,N,dict(BASE=BASE,SP=BASE/'spatial',verify=check))()
def features():
    import torch
    from scripts import run_stvg_spatial_controls256_native_v1 as original
    from scripts.stvg_native_instance_common_v1 import guard,write
    verify();sys.addaudithook(guard);original.BASE=BASE;m,p,device,proof,secs=original.load();records=[];jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records']
    for j in jobs:
        if time.time()>=c.deadline():raise TimeoutError('bounded confirmation feature checkpoint')
        b,ps,e,pos,f,runs,pol,v,ma,gr,q,caption,available=original.prepare(m,p,device,j);original.cache_write(j,proof,ps,f,pol,v,ma,gr,q,caption,available);path=BASE/'native_cache'/j['job_id']/'POLICIES.json';records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),case_id=j['case_id']));del b,e,pos,f,runs,pol;gc.collect();torch.cuda.empty_cache();write(BASE/'FEATURE_STATUS.json',dict(status='running',done=len(records),total=len(jobs),count=N,GT_read=False),mutable=True)
    put(BASE/'FEATURE_BARRIER.json',dict(status='complete',count=N,new_encoder_calls=len(records),files=records,model_load_seconds=secs,GT_read=False,qualification_status='native/generation qualification follows in each actual evidence arm'))
def rgb():
    import numpy as np
    from PIL import Image
    from scripts.stvg_native_instance_common_v1 import guard,write
    from scripts.run_stvg_weightfree_expand_v1 import selected_rgb
    from scripts.stvg_evidence_spatial_inputs005 import _letterbox
    verify();sys.addaudithook(guard);records=[]
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias'];rp=BASE/'temporal/requests'/(cid+'.json');req=read(rp);rows=sorted(req['frame_map'],key=lambda x:int(x['local_frame_index']));ids=[int(x['physical_frame_id']) for x in rows];assert len(ids)==len(set(ids))
        if len(ids)<32:records.append(dict(case_id=cid,dataset=r['dataset'],status='unavailable_less_than32_distinct_input_frames',GT_read=False));continue
        ix=np.rint(np.linspace(0,len(rows)-1,32)).astype(int);assert len(set(ix))==32;chosen=[rows[int(i)] for i in ix];fids=[int(x['physical_frame_id']) for x in chosen];times=[float(x['source_timestamp_s']) for x in chosen];assert all(b>a for a,b in zip(times,times[1:]));assert sha(Path(req['video_path']))==req['canonical_video_sha256'];images=selected_rgb(req,set(fids));z=np.ascontiguousarray(np.stack([np.asarray(_letterbox(Image.fromarray(images[f]))[0],dtype=np.uint8) for f in fids]).transpose(0,3,1,2));p=BASE/'global_inputs'/(cid+'.npy');p.parent.mkdir(exist_ok=True);assert not p.exists();np.save(p,z,allow_pickle=False);records.append(dict(case_id=cid,dataset=r['dataset'],status='ready',path=str(p.relative_to(ROOT)),sha256=sha(p),tensor_sha256=hashlib.sha256(z.tobytes()).hexdigest(),timestamps_seconds=times,physical_frame_ids=fids,GT_read=False));del z,images;gc.collect();write(BASE/'RGB_STATUS.json',dict(status='running',done=len(records),total=N,count=N,GT_read=False),mutable=True)
    put(BASE/'GLOBAL_RGB_BARRIER.json',dict(status='complete',count=N,records=records,GT_read=False));put(BASE/'ROOT_RGB_READBACK.json',dict(status='pass',count=N,barrier_sha256=sha(BASE/'GLOBAL_RGB_BARRIER.json'),GT_read=False))
def scopes():
    verify();candidate=read(c.BASE/'CANDIDATE_LOCK.json');assert candidate['confirmation_roster_sha256']==sha(BASE/'ROSTER_LOCK.json');global_candidate=candidate['selected_scope']=='global_B';arms=[('A','A',global_candidate),('B','B',global_candidate)]+([('core_B','B',False)] if global_candidate else [])
    for label,policy,isglobal in arms:
        b=BASE/'arms'/label;b.mkdir(parents=True,exist_ok=False);(b/'ROSTER_LOCK.json').write_bytes((BASE/'ROSTER_LOCK.json').read_bytes())
        if isglobal:
            for name in ('GLOBAL_RGB_BARRIER.json','ROOT_RGB_READBACK.json'):(b/name).write_bytes((BASE/name).read_bytes())
        cfg=dict(label='locked confirmation128 '+label,core_scope=str(BASE.relative_to(ROOT)),design_template=candidate['design_template'],policy=policy,recipe=candidate['recipe'],global_frames=32 if isglobal else 4,global_keep=512 if isglobal else 64,local_keep=96,physical_clock=isglobal,vision_reuse_scope=str((BASE/'arms/A').relative_to(ROOT)) if policy=='B' and isglobal else None,ablation=None,local_control='B',count=N,GT_read=False);put(b/'SCOPE_CONFIG.json',cfg)
    put(BASE/'ARM_LOCK.json',dict(status='locked',count=N,arms=[x[0] for x in arms],all_arms_seal_before_any_GT=True,candidate_sha256=sha(c.BASE/'CANDIDATE_LOCK.json'),GT_read=False))
def contracts():
    from scripts import prepare_stvg_spatial_controls256_v1 as source,run_stvg_spatial_controls256_temporal_v1 as temporal,run_stvg_spatial_controls256_inputs_v1 as spatial,stvg_instance_temporal32_v1 as fit
    # Count adaptation is confined to outer cohort functions; 32 Gaussian samples,
    # 512 Temporal / 1792 Spatial parameters and 32 global frames stay untouched.
    for fn,a in ((source.source,256),(temporal.prepare,256),(spatial.prepare,256),(spatial.input_root,256),(fit.teacher,32),(fit.fit_cases,32)):
        transform(fn,a,128,{})
    print('CONFIRMATION128_SIX_DISPATCH_AST_CONTRACTS_PASS_NO_MODEL_NO_GT')
if __name__=='__main__':
    mode=sys.argv[1]
    if mode.startswith('temporal_'):temporal(mode.split('_',1)[1])
    elif mode.startswith('spatial_'):spatial(mode.split('_',1)[1])
    else:{'setup':setup,'source':source,'features':features,'rgb':rgb,'scopes':scopes,'contracts':contracts}[mode]()
