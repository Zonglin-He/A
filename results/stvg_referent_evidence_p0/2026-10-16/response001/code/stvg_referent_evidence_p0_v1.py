"""Bounded precondition study; historical studies and model sources stay immutable.

Source geometry control changes only the adopted reference coordinates. The
separate evidence diagnostic uses native projected Teacher tokens, never GT to
select tokens. Its connectivity is a feature correspondence proxy, not CS.
"""
from pathlib import Path
import ast,copy,gc,hashlib,inspect,json,os,subprocess,sys,threading,time,traceback,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE as PRIOR,read,put,sha,guard
BASE=ROOT/'artifacts/stvg_referent_evidence_p0_v1';CONTROL=BASE/'source_control'
UNIT='stvg-referent-evidence-p0-001.service'
def setup():
    import torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='complete'
    rr=read(PRIOR/'ROSTER_LOCK.json')['records'];jobs=read(PRIOR/'temporal/INPUT_BARRIER.json')['jobs'];available={read(ROOT/j['path'])['case_id']for j in jobs}
    chosen=[]
    for ds in ('hc2','vidstg'):
        chosen+=sorted([r for r in rr if r['dataset']==ds and r['probe_alias']in available],key=lambda r:(r['parent_sha256'],r['query_key_sha256']))[:8]
    assert len(chosen)==16 and len({r['parent_sha256']for r in chosen})==16
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=16,records=chosen,selection='first eight parent/query SHA per domain among prior development32 with available original candidate clips; no result selection',all_development_exposed=True,GT_read=False))
    design=dict(status='locked',phase='mechanism_before_method',count=16,source_control_count=32,
        fixed='original corrected-clock native R1, exact marked eight448 candidate clips, original query and already sealed caption; no new caption generation or Prompt search; Student optimizer unchanged',
        source_control='replace ONLY coordinates of every adopted E1 reference with original Source box at that identical position; preserve E1 admission/positions/presence/weights, actual 1792 M2 fullstep/full_grid re-decode; diagnostic self-reference, not new independent evidence',
        response='native frozen visual encoder+merger tokens vs native input text embedding rows, mean top-two cosine similarities following released Qwen2.5 SemVID query reduction; percentile ranks per temporal cell; U=min(Q,C), D=abs(Q-C)',
        strategies=['query','caption','consensus','disagreement','connection'],retention_ratio=.125,
        connectivity='mutual nearest projected-feature correspondence within normalized 0.25 image distance, adjacent merged temporal cells; skip transitions whose normalized global-mean cosine is below .5; identity proxy only, not attention routing/CS or verified object identity; uncertain edges not forced',
        connection_selection='equal per-cell budget; half consensus anchors, remainder highest mutual-edge scores with geometric proximity; no new object box derived from token selections',
        mapping='each post-merger token spans a 28x28 displayed cell and TWO original physical frames; preserve resize letterboxing and frame pair, do not label merged tokens as single-frame',
        additional_backbone_responses=False,attention_ER_CS=False,training_free=True,bootstrap=False,CURRENT_change=False,official_fullset=False,
        resource_budget='at most 64 native visual encoder passes, zero new caption/alignment generation; 32 one-method actual Student controls; 90 minutes wall budget for this stage, fail visibly rather than expand',
        phase2_gate='both domains must show positive source-macro consensus-minus-best-single target-area occupancy on the SAME known positions and positive connection-minus-query adjacent supported pair coverage; at least four known sources/domain; no automatic gate tuning. If not supported, report phase1 and do not run A/B/C.',
        phase2='conditional only: official fixed SemVID A vs dual-text constrained B vs B+connection C, equal per-frame quotas/input/call/token budgets and fixed OPD; requires separately qualified native pruning insertion, never substitute RGB masking and call it token pruning',
        GT_read=False)
    put(BASE/'DESIGN_LOCK.json',design)
    CONTROL.mkdir(exist_ok=False)
    for name in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json'):(CONTROL/name).write_bytes((PRIOR/name).read_bytes())
    b=CONTROL/'trials/SOURCE';b.mkdir(parents=True)
    refs=copy.deepcopy(read(PRIOR/'trials/E1/REFERENCE_LOCK.json'));src={}
    for rec in rr:
        c,p=load_case(rec);src[rec['probe_alias']]=c['native_boxes_xyxy_normalized'];del c;gc.collect()
    changed=0
    for r in refs['records']:
        for a in r['anchors']:
            # The original anchor coordinate field is validated, not guessed.
            assert 'box_cxcywh' in a,sorted(a)
            x1,y1,x2,y2=map(float,src[r['case_id']][a['position']])
            a['box_cxcywh']=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1];changed+=1
    refs.update(scope='Source-coordinate matched-support control; E1 selection/weights unchanged',GT_read=False)
    put(b/'REFERENCE_LOCK.json',refs);(b/'TEMPORAL_LOCK.json').write_bytes((PRIOR/'trials/E1/TEMPORAL_LOCK.json').read_bytes())
    d=copy.deepcopy(read(PRIOR/'trials/E0/DESIGN_LOCK.json'));d.update(trial_id='SOURCE',scope=design['source_control'])
    put(b/'DESIGN_LOCK.json',d)
    rt=copy.deepcopy(read(PRIOR/'trials/E0/RUNTIME.json'));rt.update(trial_id='SOURCE',scope=design['source_control'])
    pins=[Path(__file__),BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',CONTROL/'ROSTER_LOCK.json',CONTROL/'SOURCE_CAPTURE_BARRIER.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json',ROOT/'scripts/review_stvg_instance_student32_002.py']
    for p in pins:rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(b/'RUNTIME.json',rt)
    selected={r['probe_alias']for r in chosen};jj=[]
    teacherfiles={x['path']:x['sha256']for x in read(PRIOR/'temporal/TEACHER_GLOBAL_BARRIER.json')['files']}
    for j in jobs:
        m=read(ROOT/j['path']);cid=m['case_id']
        if cid not in selected:continue
        p=PRIOR/'temporal/teacher'/cid/(m['candidate']['canonical_id']+'.json');assert sha(p)==teacherfiles[str(p.relative_to(ROOT))]
        assert sha(ROOT/m['input_path'])==m['input_sha256']
        req=PRIOR/'temporal/requests'/(cid+'.json')
        jj.append(dict(metadata_path=j['path'],metadata_sha256=j['sha256'],caption_path=str(p.relative_to(ROOT)),caption_sha256=sha(p),request_sha256=sha(req),case_id=cid))
    assert len(jj)<=64
    put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=16,jobs=jj,GT_read=False,source_control_reference_count=changed))
    put(BASE/'RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'INPUT_BARRIER.json',ROOT/'scripts/stvg_referent_response_v1.py']},GT_read=False))
    put(BASE/'RESPONSE_RUNTIME.json',dict(status='locked',pins=read(BASE/'RUNTIME.json')['pins'],GT_read=False))
    put(BASE/'STATUS.json',dict(status='prepared',stage='source_control_then_native_response',count=16,GT_read=False))
    print(json.dumps(dict(status='locked',sources=16,jobs=len(jj),control_references=changed)))
def verify(tid=None):
    if tid:
        rt=read(CONTROL/'trials'/tid/'RUNTIME.json');pins={**rt['pins'],**rt['protected']}
    else:pins=read(BASE/'RUNTIME.json')['pins']
    for rel,h in pins.items():assert sha(ROOT/rel)==h,rel
def control(mode):
    from scripts import run_stvg_instance_student32_v1 as old
    old.BASE=CONTROL;old.check=lambda tid:verify(tid);old.modes=lambda tid:('M2',)
    if mode=='worker':old.worker('SOURCE')
    else:
        # Same independently saved arithmetic reader, matching float64 readout.
        code=inspect.getsource(old.root);assert code.count("bb=arr(o['boxes']);xy=")==1
        code=code.replace("bb=arr(o['boxes']);xy=","bb=arr(o['boxes']).astype(float);xy=")
        ns=dict(old.__dict__);exec(compile(code,str(Path(__file__)),'exec'),ns);ns['root']('SOURCE')
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as r
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='source_control';g=1024**3;v=r.sample()
    assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start();start=time.monotonic()
    try:
        for mode in ('control_worker','control_root','responses'):
            assert time.monotonic()-start<5400
            p=BASE/'execution'/mode;p.mkdir(parents=True,exist_ok=False)
            interp='.venv/stvg_native_distill_v1/bin/python'if mode=='responses'else'.conda/tubedetr/bin/python'
            with(p/'worker.log').open('xb')as log:
                t=time.perf_counter();c=subprocess.Popen([str(ROOT/interp),'-B',str(Path(__file__)),mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
                put(p/'LAUNCH.json',dict(status='running',pid=c.pid,GT_read=False));rc=c.wait(timeout=max(1,5400-(time.monotonic()-start)))
            put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,mode
    finally:stop.set()
def responses():
    from scripts.stvg_referent_response_v1 import main
    main()
if __name__=='__main__':
    try:
        mode=sys.argv[1]
        if mode.startswith('control_'):control(mode.split('_')[1])
        else:{'setup':setup,'supervise':supervise,'responses':responses}[mode]()
    except BaseException as e:
        p=BASE/('FAILURE_'+sys.argv[1]+'.json')
        if not p.exists():put(p,dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False))
        raise
