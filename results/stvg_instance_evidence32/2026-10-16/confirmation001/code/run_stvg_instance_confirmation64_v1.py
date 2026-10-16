"""One frozen E1 recipe on the pre-existing disjoint64; no retuning.

Adapters change only cohort cardinality/layout and the already qualified
float64 saved-box reader. Model/feedback/fitter/complete readout stay original.
"""
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,threading,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE as DEV,LITE,read,put,sha
BASE=DEV/'confirmation';UNIT='stvg-instance-evidence64-confirm-001.service'
SOURCES=['stvg_instance_temporal32_v1.py','stvg_instance_spatial32_v1.py','run_stvg_instance_teacher32_v1.py','run_stvg_instance_student32_v1.py','review_stvg_instance_fit_binding_v1.py','score_stvg_instance_experiment32_v1.py','score_stvg_instance32_v1.py']
def setup():
    decision=read(DEV/'DEVELOPMENT_DECISION.json');assert decision['status']=='locked_for_confirmation'and decision['selected_recipe']=='E1'
    assert not BASE.exists();BASE.mkdir()
    r=read(LITE/'confirmation/ROSTER_LOCK.json');rr=r['records'];assert len(rr)==64 and len({x['parent_sha256']for x in rr})==64
    assert not {x['parent_sha256']for x in rr}&{x['parent_sha256']for x in read(DEV/'ROSTER_LOCK.json')['records']}
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=64,records=rr,GT_read=False,scope='same fixed old64 confirmation parents, disjoint current32; previously exposed to historical project results, not globally blind'))
    (BASE/'EVIDENCE_MODIFICATION_LOCK.json').write_bytes((DEV/'EVIDENCE_MODIFICATION_LOCK.json').read_bytes())
    put(BASE/'FROZEN_RECIPE_LOCK.json',dict(status='locked',count=64,selected_recipe='E1',development_decision_sha256=sha(DEV/'DEVELOPMENT_DECISION.json'),design_sha256=sha(DEV/'trials/E1/DESIGN_LOCK.json'),same_params_same_prompts_same_candidate_rule=True,comparators=['Frozen','corrected Temporal','E0 old evidence/current updater','E1 new evidence/current updater','E2 same E1 references/original validated OPD'],no_confirmation_selection=True,GT_read=False))
    pins={str(ROOT/'scripts'/s):sha(ROOT/'scripts'/s)for s in SOURCES+[Path(__file__).name]}
    pins={str(Path(k).relative_to(ROOT)):v for k,v in pins.items()}
    for p in [BASE/'ROSTER_LOCK.json',BASE/'FROZEN_RECIPE_LOCK.json',BASE/'EVIDENCE_MODIFICATION_LOCK.json',DEV/'DEVELOPMENT_DECISION.json',LITE/'confirmation/TEMPORAL_GLOBAL_BARRIER.json',LITE/'confirmation/ROSTER_LOCK.json']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    put(BASE/'RUNTIME.json',dict(status='locked',pins=pins,GT_read=False,scope='only fixed64 frozen recipe confirmation; actual original Temporal baseline must replay exactly before corrected feedback'))
def verify():
    for rel,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
def module(name,extra=None):
    verify();p=ROOT/'scripts'/name;source=p.read_text()
    if name=='stvg_instance_temporal32_v1.py':
        for a,b in [("LITE/'reclocked_inputs/requests'","LITE/'confirmation/requests'"),("LITE/'temporal/revision003/student'","LITE/'confirmation/temporal'"),("d['fits']['temporal_lite']","d['selected_temporal_fit']"),("d['arms']['TS_temporal_lite']['interval']","d['selected_interval']")]:
            assert source.count(a)==1,(name,a);source=source.replace(a,b)
    if name=='stvg_instance_spatial32_v1.py':
        a="LITE/'reclocked_inputs/new_crop/proposals'";assert source.count(a)==1;source=source.replace(a,"LITE/'confirmation/temporal_inputs/new_crop/proposals'")
    if name=='run_stvg_instance_student32_v1.py':
        a="bb=arr(o['boxes']);xy=";assert source.count(a)==1;source=source.replace(a,"bb=arr(o['boxes']).astype(float);xy=")
    tree=ast.parse(source)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value)is int and n.value==32:return ast.copy_location(ast.Constant(64),n)
            if name=='score_stvg_instance32_v1.py'and type(n.value)is int and n.value==16:return ast.copy_location(ast.Constant(32),n)
            return n
    tree=ast.fix_missing_locations(Count().visit(tree));m=types.ModuleType('scripts._confirm64_'+p.stem);m.__dict__.update(__name__=m.__name__,__file__=str(p));exec(compile(tree,str(p),'exec'),m.__dict__)
    m.BASE=BASE;m.OUT=BASE/'temporal';m.SP=BASE/'spatial';m.verify=verify;m.UNIT=UNIT
    if name=='stvg_instance_temporal32_v1.py':m.context=lambda rec:2
    if name=='score_stvg_instance_experiment32_v1.py':m.aggregate=module('score_stvg_instance32_v1.py').aggregate
    return m
def validate_original_temporal():
    bar=read(LITE/'confirmation/TEMPORAL_GLOBAL_BARRIER.json');assert bar['count']==64 and bar['status']=='complete'and not bar['GT_read']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
def stage(category,action):
    verify()
    if category=='temporal':
        m=module('stvg_instance_temporal32_v1.py')
        if action=='prepare':validate_original_temporal();m.prepare()
        elif action in ('fit','readback'):m.fit_cases(action=='readback')
        else:m.teacher()
    elif category=='spatial':
        m=module('stvg_instance_spatial32_v1.py');{'prepare':m.prepare,'root':m.input_root}[action]()
    elif category=='teacher':
        m=module('run_stvg_instance_teacher32_v1.py');{'prepare':m.prepare,'worker':m.worker,'root':m.root}[action]()
    elif category=='student':
        m=module('run_stvg_instance_student32_v1.py')
        if action=='prepare':m.prepare()
        else:
            mode,tid=action.split(':');{'worker':m.worker,'root':m.root}[mode](tid)
    elif category=='seal':
        put(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,trials={t:sha(BASE/'trials'/t/'GLOBAL_PREDICTION_BARRIER.json')for t in ('E0','E1')},GT_read=False,confirmation_recipe_sha256=sha(BASE/'FROZEN_RECIPE_LOCK.json')))
    elif category=='evaluate':
        module('review_stvg_instance_fit_binding_v1.py').main();module('score_stvg_instance_experiment32_v1.py').main()
    else:raise ValueError(category)
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='frozen64_confirmation';g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x);assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    stages=[('temporal',x)for x in ('prepare','teacher','fit','readback')]+[('spatial',x)for x in ('prepare','root')]+[('teacher',x)for x in ('prepare','worker','root')]+[('student','prepare')]+[('student',mode+':'+tid)for tid in ('E0','E1')for mode in ('worker','root')]+[('seal','all'),('evaluate','all')]
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for i,(category,action)in enumerate(stages):
            p=BASE/'execution'/f'{i:02}_{category}_{action.replace(":","_")}';p.mkdir(parents=True,exist_ok=False)
            interp=ROOT/('.venv/stvg_native_distill_v1/bin/python'if(category,action)in [('temporal','teacher'),('teacher','worker')]else'.conda/tubedetr/bin/python')
            from scripts.stvg_native_instance_common_v1 import write
            write(BASE/'STATUS.json',dict(status='running',stage=category+'_'+action,done=i,total=len(stages),GT_read=False),mutable=True)
            with(p/'worker.log').open('xb')as log:
                t=time.perf_counter();c=subprocess.Popen([str(interp),'-B',str(Path(__file__)),'stage',category,action],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(p/'LAUNCH.json',dict(status='running',pid=c.pid,GT_read=False));code=c.wait()
            put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=category=='evaluate'));assert code==0,(category,action)
        put(BASE/'MEASUREMENT_COMPLETION.json',dict(status='complete_pending_independent_public_archive',count=64,GT_read=True,result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json')))
        write(BASE/'STATUS.json',dict(status='complete_pending_independent_public_archive',stage='root_handoff',count=64),mutable=True)
    finally:stop.set()
if __name__=='__main__':
    if sys.argv[1]=='setup':setup()
    elif sys.argv[1]=='stage':stage(*sys.argv[2:4])
    else:supervise()
