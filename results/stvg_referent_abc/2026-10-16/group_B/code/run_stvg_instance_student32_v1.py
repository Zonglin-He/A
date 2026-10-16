"""Fixed E0/E1/E2: unchanged complete-model Student and saved-math audits."""
from pathlib import Path
import ast,copy,gc,json,os,subprocess,sys,threading,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,LITE,read,put,sha,verify
SP=BASE/'spatial';UNIT='stvg-instance-evidence32-student-001.service'
def modes(tid):return ('M2',)if tid=='E0'else('M0','M2')
def prepare():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    verify();assert read(SP/'ROOT_TEACHER_READBACK.json')['status']=='pass'
    jobs=read(SP/'TEACHER_JOB_LOCK.json')['records'];source=[];rr=read(BASE/'ROSTER_LOCK.json')['records']
    for rec in rr:
        c,p=load_case(rec);source.append(dict(case_id=rec['probe_alias'],path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,source_case_sha256=c['source_case_sha256']));del c;gc.collect()
    put(BASE/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=32,files=source,GT_read=False))
    oldrt=read(LITE/'trials/C_FULL/RUNTIME.json');old=read(LITE/'trials/C_FULL/DESIGN_LOCK.json')
    for tid in ('E0','E1'):
        b=BASE/'trials'/tid;b.mkdir(parents=True,exist_ok=False);refs=[]
        for rec in rr:
            cid=rec['probe_alias'];c,p=load_case(rec);pool=read(SP/tid/'proposals'/f'{cid}.json');anchors=[];decisions=[]
            for j in [j for j in jobs if j['evidence_group']==tid and j['case_id']==cid]:
                rp=SP/'teacher'/f"{j['job_id']}.json";response=read(rp);w=next(w for w in pool['windows']if w['slot']==j['center_slot'])
                a,why=_anchor_from_teacher_selection(j,response,c['frame_ids'],w['source_grid_position'],teacher_output_sha256=sha(rp))
                decisions.append(dict(slot=j['center_slot'],reason=why,selected_candidate_id=response['parsed'].get('selected_candidate_id'),teacher_sha256=sha(rp),GT_read=False))
                if a is not None:anchors.append(a)
            assert len({a['position']for a in anchors})==len(anchors)
            refs.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=c['source_case_sha256'],anchors=anchors,decisions=decisions,GT_read=False));del c;gc.collect()
        put(b/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=refs,GT_read=False,teacher_barrier_sha256=sha(SP/'TEACHER_GLOBAL_BARRIER.json'),source_candidate_forbidden=True,presence_threshold=.6))
        (b/'TEMPORAL_LOCK.json').write_bytes((BASE/'temporal/TEMPORAL_LOCK.json').read_bytes())
        d=copy.deepcopy(old);d.update(trial_id=tid,cohort='development32',scope='one fixed instance hypothesis; E0 old evidence/M2, E1 new evidence/M2 and same-reference M0=E2',GT_read=False)
        for ds in d['configs']:
            d['configs'][ds]={m:d['configs'][ds][m]for m in modes(tid)};d['family_reuse_from'][ds]={m:None for m in modes(tid)}
        d.pop('recipe_sha256',None);d['evidence_design_sha256']=sha(BASE/'EVIDENCE_MODIFICATION_LOCK.json');put(b/'DESIGN_LOCK.json',d)
        pins=copy.deepcopy(oldrt['pins']);protected=copy.deepcopy(oldrt['protected'])
        for p in [Path(__file__),ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py',ROOT/'scripts/audit_stvg_spatial_lite_fit002.py',ROOT/'scripts/review_stvg_lite_complete002.py',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'ROSTER_LOCK.json',BASE/'EVIDENCE_MODIFICATION_LOCK.json',SP/'ROOT_TEACHER_READBACK.json',SP/'TEACHER_GLOBAL_BARRIER.json',b/'DESIGN_LOCK.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json']:
            pins[str(p.relative_to(ROOT))]=sha(p)
        for rel,h in {**pins,**protected}.items():assert sha(ROOT/rel)==h,rel
        put(b/'RUNTIME.json',dict(status='locked',trial_id=tid,pins=pins,protected=protected,GT_read=False))
    put(BASE/'STUDENT_EXECUTION_LOCK.json',dict(status='locked',count=32,trials=['E0','E1'],E2='E1/M0; exactly same reference document as E1/M2',GT_read=False))
def check(tid):
    verify();rt=read(BASE/'trials'/tid/'RUNTIME.json')
    for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
def worker(tid):
    import torch
    from scripts import run_stvg_soft_ep_coordinate_trial003 as original
    from scripts import stvg_soft_ep_coordinate_fit002 as dispatch
    from scripts import stvg_spatial_lite_fit_v1 as lite
    check(tid);tree=ast.parse(Path(original.__file__).read_text())
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(32),n)if type(n.value)is int and n.value==128 else n
    tree=ast.fix_missing_locations(Count().visit(tree));w=types.ModuleType('scripts._instance_evidence_complete_worker');w.__dict__.update(__name__=w.__name__,__file__=original.__file__);exec(compile(tree,original.__file__,'exec'),w.__dict__)
    w.BASE=BASE;w.TRIALS=BASE/'trials';w.ROSTER_PATH=BASE/'ROSTER_LOCK.json';w.SOURCE_BARRIER_PATH=BASE/'SOURCE_CAPTURE_BARRIER.json';w.verify_runtime=lambda:check(tid);w.MODES=modes(tid);w.ARM_NAMES=('F','T')+tuple(x+m for m in modes(tid)for x in ('S_','TS_'))
    old=dispatch.fit
    def fitter(base,initial,anchors,ids,key,cfg,mode,settings,qualification=False,checkpoint=None):
        torch.cuda.synchronize();tick=time.perf_counter()
        f=lite.fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=qualification,checkpoint=checkpoint,recipe='lite_fullstep',response_scope='full_grid',event_support_positions=None)
        torch.cuda.synchronize();f['instance_study_fit_seconds']=time.perf_counter()-tick;return f
    try:dispatch.fit=fitter;sys.argv=['instance_student',tid];w.main()
    finally:dispatch.fit=old
def root(tid):
    import torch,numpy as np
    from scripts.audit_stvg_soft_ep_v1 import arr,treeeq
    from scripts.audit_stvg_spatial_lite_fit002 import fit_audit
    from scripts.review_stvg_lite_complete002 import audit_m0
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    check(tid);b=BASE/'trials'/tid;bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==32 and not bar['GT_read'];dsg=read(b/'DESIGN_LOCK.json')
    refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};tt={r['case_id']:r for r in read(b/'TEMPORAL_LOCK.json')['records']};rr={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};seen=set();counts={m:dict(accepted=0,proposals=0,no_reference=0,actual_fit=0)for m in modes(tid)}
    for e in bar['files']:
        p=ROOT/e['path'];assert sha(p)==e['sha256']and sha(ROOT/e['receipt_path'])==e['receipt_sha256'];receipt=read(ROOT/e['receipt_path']);assert receipt['sha256']==sha(p)and receipt['status']=='complete'and not receipt['GT_read']
        d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert cid not in seen;seen.add(cid);rec=rr[cid];c,cp=load_case(rec);ref=refs[cid]
        assert d['runtime_sha256']==sha(b/'RUNTIME.json')and d['source_restored']and not d['GT_read']and d['source_file_sha256']==sha(cp)
        assert d['reference_records']==ref and d['temporal_record']==tt[cid]
        assert np.array_equal(d['arms']['F']['boxes'],c['native_boxes_xyxy_normalized'])and np.array_equal(d['arms']['T']['boxes'],d['arms']['F']['boxes'])
        assert d['arms']['F']['interval']==c['Native_interval']and d['arms']['T']['interval']==tt[cid]['selected_interval']
        for m in modes(tid):
            f=d['fits'][m];o=d['readouts'][m]
            if not ref['anchors']:
                assert f is o is None;counts[m]['no_reference']+=1
                for pre in ('S_','TS_'):assert np.array_equal(d['arms'][pre+m]['boxes'],d['arms']['F']['boxes'])
                continue
            assert treeeq(f['initial_state'],c['source_light_initial_state']);cfg=dsg['configs'][rec['dataset']][m];key=stable_fit_key(rec['dataset'],rec['query_key_sha256'])
            a=audit_m0(f,ref,key,cfg)if m=='M0'else fit_audit(f,ref,'lite_fullstep',response_positions=None)
            for k in ('accepted','proposals'):counts[m][k]+=a[k]
            counts[m]['actual_fit']+=1;assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu());bb=arr(o['boxes']);xy=np.concatenate([bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2],axis=1)
            for pre in ('S_','TS_'):assert np.array_equal(xy,d['arms'][pre+m]['boxes'])
            assert d['arms']['S_'+m]['interval']==c['Native_interval']and d['arms']['TS_'+m]['interval']==tt[cid]['selected_interval']
        del c,d;gc.collect()
    assert len(seen)==len(bar['files'])==32
    put(b/'ROOT_READBACK.json',dict(status='pass',count=32,counters=counts,GT_read=False,CPU_only=True,barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),scope='independent full saved rollout/Adam/response solve/acceptance and actual complete-model readout; no separate full backward'))
    put(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=bar['files'],GT_read=False,root_sha256=sha(b/'ROOT_READBACK.json')))
    print('INSTANCE_STUDENT_ROOT_PASS',tid,counts,flush=True)
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='student';g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for tid in ('E0','E1'):
            for mode in ('worker','root'):
                p=BASE/'student_execution'/tid/mode;p.mkdir(parents=True,exist_ok=False)
                with(p/'worker.log').open('xb')as log:
                    t=time.perf_counter();c=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),mode,tid],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT);put(p/'LAUNCH.json',dict(status='running',pid=c.pid,GT_read=False));code=c.wait()
                put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=False));assert code==0,(tid,mode)
        put(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,trials={tid:sha(BASE/'trials'/tid/'GLOBAL_PREDICTION_BARRIER.json')for tid in ('E0','E1')},GT_read=False))
    finally:stop.set()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('prepare','supervise'):{'prepare':prepare,'supervise':supervise}[mode]()
    else:{'worker':worker,'root':root}[mode](sys.argv[2])
