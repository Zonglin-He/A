"""Fixed final Lite Student re-decode only where corrected WHO geometry changes.

Exact unchanged geometry/source/parameters reuse is explicit; no second GPU fit
is counted for those cases. The comparison preserves historical Temporal output.
"""
from pathlib import Path
import ast,copy,gc,hashlib,json,os,subprocess,sys,threading,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_geometry_audit_v1 import BASE as STUDY,LITE,SEARCH,read,put,sha
BASE=STUDY/'clock_control';UNIT='stvg-reference-clock-student-001.service';TID='C_CLOCK'

def geometry(ref):return [{k:v for k,v in a.items()if k not in ('teacher_output_sha256','GT_read')}for a in ref['anchors']]

def prepare():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    assert read(BASE/'ROOT_TEACHER_READBACK.json')['status']=='pass'
    jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];roster=read(STUDY/'ROSTER_LOCK.json')['records'];oldrefs={r['case_id']:r for r in read(LITE/'trials/C_FULL/REFERENCE_LOCK.json')['records']};references=[];changed=[]
    for rec in roster:
        cid=rec['probe_alias'];case,cp=load_case(rec);anchors=[]
        for job in jobs:
            if job['case_id']!=cid:continue
            p=BASE/'teacher'/f"{job['job_id']}.json";response=read(p);pos=list(map(int,case['frame_ids'])).index(job['center_physical_frame_id'])
            anchor,reason=_anchor_from_teacher_selection(job,response,case['frame_ids'],pos,teacher_output_sha256=sha(p))
            if anchor is not None:anchors.append(anchor)
        ref=dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],anchors=anchors,GT_read=False)
        references.append(ref)
        if geometry(ref)!=geometry(oldrefs[cid]):changed.append(rec)
        del case;gc.collect()
    put(BASE/'REFERENCE_LOCK.json',dict(status='locked',count=12,records=references,GT_read=False))
    put(BASE/'CHANGED_SOURCE_LOCK.json',dict(status='locked',count=len(changed),records=changed,GT_read=False,selection='exact geometry equality only; no GT or metric threshold'))
    assert len(changed)==1 and changed[0]['probe_alias']=='HC01','finite control observed a different changed set; inspect before GPU'
    # The assertion documents the actually completed WHO result, not a method
    # rule. This finite one-case worker must never expand itself silently.
    ids={r['probe_alias']for r in changed};trial=BASE/'trials'/TID;trial.mkdir(parents=True,exist_ok=False)
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=len(changed),records=changed,GT_read=False))
    source=read(SEARCH/'SOURCE_CAPTURE_BARRIER.json');source['files']=[r for r in source['files']if r.get('case_id')in ids or any(r['path']==c['source_case_path']for c in changed)];source['count']=len(changed);assert len(source['files'])==len(changed)
    put(BASE/'SOURCE_CAPTURE_BARRIER.json',source)
    put(trial/'REFERENCE_LOCK.json',dict(status='locked',count=len(changed),records=[r for r in references if r['case_id']in ids],GT_read=False))
    temporal=read(LITE/'trials/C_FULL/TEMPORAL_LOCK.json');temporal['records']=[r for r in temporal['records']if r['case_id']in ids];temporal['count']=len(changed);put(trial/'TEMPORAL_LOCK.json',temporal)
    design=read(LITE/'trials/C_FULL/DESIGN_LOCK.json');design['trial_id']=TID;design['scope']='one exact changed-reference source, fixed final Lite; no parameter selection'
    for ds in ('hc2','vidstg'):
        design['configs'][ds]={m:v for m,v in design['configs'][ds].items()if m=='M2'};design['family_reuse_from'][ds]={'M2':None}
    put(trial/'DESIGN_LOCK.json',design)
    runtime=read(LITE/'trials/C_FULL/RUNTIME.json');runtime['trial_id']=TID
    for p in [Path(__file__),ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',BASE/'ROOT_TEACHER_READBACK.json',BASE/'REFERENCE_LOCK.json',BASE/'CHANGED_SOURCE_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',trial/'REFERENCE_LOCK.json',trial/'TEMPORAL_LOCK.json',trial/'DESIGN_LOCK.json']:
        runtime['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(trial/'RUNTIME.json',runtime)
    print(json.dumps({'status':'locked','changed_references':len(changed),'exact_geometry_unchanged':12-len(changed)}))

def verify():
    r=read(BASE/'trials'/TID/'RUNTIME.json')
    for rel,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/rel)==h,rel
    return r

def worker():
    import torch
    from scripts import run_stvg_soft_ep_coordinate_trial003 as original
    from scripts import stvg_soft_ep_coordinate_fit002 as dispatch
    from scripts import stvg_spatial_lite_fit_v1 as lite
    source=Path(original.__file__);tree=ast.parse(source.read_text())
    class Cardinality(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value)is int and node.value==128:return ast.copy_location(ast.Constant(1),node)
            return node
    tree=ast.fix_missing_locations(Cardinality().visit(tree));ns=dict(__name__='scripts._reference_clock_student',__file__=str(source));exec(compile(tree,str(source),'exec'),ns)
    ns.update(BASE=BASE,TRIALS=BASE/'trials',ROSTER_PATH=BASE/'ROSTER_LOCK.json',SOURCE_BARRIER_PATH=BASE/'SOURCE_CAPTURE_BARRIER.json',verify_runtime=verify,MODES=('M2',),ARM_NAMES=('F','T','S_M2','TS_M2'))
    def fitter(base,initial,anchors,ids,key,cfg,mode,settings,qualification=False,checkpoint=None):
        return lite.fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=qualification,checkpoint=checkpoint,recipe='lite_fullstep',response_scope='full_grid',event_support_positions=None)
    old=dispatch.fit
    try:
        dispatch.fit=fitter;sys.argv=['reference_clock_student',TID];ns['main']()
    finally:dispatch.fit=old

def root():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.audit_stvg_spatial_lite_fit002 import fit_audit
    from scripts.audit_stvg_soft_ep_v1 import arr,treeeq
    from scripts.run_stvg_table1_student004 import stable_fit_key
    verify();trial=BASE/'trials'/TID;bar=read(trial/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==1 and bar['GT_read']is False
    rec=read(BASE/'ROSTER_LOCK.json')['records'][0];case,cp=load_case(rec);entry=bar['files'][0];p=ROOT/entry['path'];assert sha(p)==entry['sha256']and sha(ROOT/entry['receipt_path'])==entry['receipt_sha256']
    d=torch.load(p,map_location='cpu',weights_only=False);f=d['fits']['M2'];ref=read(trial/'REFERENCE_LOCK.json')['records'][0];design=read(trial/'DESIGN_LOCK.json')
    assert d['reference_records']==ref and d['source_restored']and not d['GT_read']and d['source_file_sha256']==sha(cp)
    assert f['config']==design['configs'][rec['dataset']]['M2']and f['seed_key']==stable_fit_key(rec['dataset'],rec['query_key_sha256'])and treeeq(f['initial_state'],case['source_light_initial_state'])
    audit=fit_audit(f,ref,'lite_fullstep',response_positions=None)
    assert torch.equal(d['readouts']['M2']['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu())
    bb=arr(d['readouts']['M2']['boxes']).astype(float);xy=np.concatenate([bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2],axis=1);assert np.array_equal(xy,d['arms']['TS_M2']['boxes'])
    assert np.array_equal(d['arms']['T']['boxes'],case['native_boxes_xyxy_normalized'])and d['arms']['T']['interval']==d['arms']['TS_M2']['interval']
    put(trial/'ROOT_READBACK.json',dict(status='pass',count=1,audit=audit,GT_read=False,CPU_only=True,barrier_sha256=sha(trial/'STUDENT_GLOBAL_BARRIER.json'),scope='saved original rollout/loss/Adam/response solver/rollback plus exact real final decoder output; not an independent backward'))
    oldbar=read(LITE/'trials/C_FULL/STUDENT_GLOBAL_BARRIER.json');oldfiles={r['case_id']:r for r in oldbar['files']};refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};oldrefs={r['case_id']:r for r in read(LITE/'trials/C_FULL/REFERENCE_LOCK.json')['records']};files=[]
    for r in read(STUDY/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias']
        if cid==rec['probe_alias']:files.append(dict(case_id=cid,path=entry['path'],sha256=entry['sha256'],new_fit=True));continue
        assert geometry(refs[cid])==geometry(oldrefs[cid]);o=oldfiles[cid];assert sha(ROOT/o['path'])==o['sha256']
        q=torch.load(ROOT/o['path'],map_location='cpu',weights_only=False);assert q['reference_records']==oldrefs[cid]and not q['GT_read']
        assert q['fits']['M2']['config']==design['configs'][r['dataset']]['M2']and q['fits']['M2']['settings']==design['response_settings'][r['dataset']]['M2']
        files.append(dict(case_id=cid,path=o['path'],sha256=o['sha256'],new_fit=False,exact_geometry_and_frozen_state_reuse=True));del q;gc.collect()
    put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=12,files=files,GT_read=False,new_full_Student_decodes=1,unchanged_reference_reuses=11,root_sha256=sha(trial/'ROOT_READBACK.json'),teacher_root_sha256=sha(BASE/'ROOT_TEACHER_READBACK.json')))
    print(json.dumps({'status':'pass','count':12,'actual_fit':1,'exact_reuse':11,'audit':audit}))

def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='student';g=1024**3
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start();p=BASE/'student_execution';p.mkdir(exist_ok=False)
    try:
        for mode in ('worker','root'):
            with(p/(mode+'.log')).open('xb')as log:
                t=time.perf_counter();child=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
                put(p/(mode+'_LAUNCH.json'),dict(status='running',pid=child.pid,GT_read=False,time=time.time()));code=child.wait()
            put(p/(mode+'_EXIT.json'),dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=False,time=time.time()));assert code==0,mode
    finally:stop.set()

if __name__=='__main__':{'prepare':prepare,'worker':worker,'root':root,'supervise':supervise}[sys.argv[1]]()
