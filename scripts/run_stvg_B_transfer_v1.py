"""Fixed B extent x guarded reachability factorial; original math untouched."""
from pathlib import Path
import ast,copy,gc,inspect,json,os,subprocess,sys,threading,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE as PRIOR,read,put,sha
BASE=ROOT/'artifacts/stvg_B_transfer_v1';ARMS=('G10','R01','GR11');UNIT='stvg-B-transfer-001.service'
def extent(box):
    x,y,w,h=map(float,box);assert 0<x<1 and 0<y<1 and 0<w<=2*min(x,1-x)+1e-7 and 0<h<=2*min(y,1-y)+1e-7
    return[x,y,min(1.25*w,2*min(x,1-x)),min(1.25*h,2*min(y,1-y))]
def recipe(arm):return'lite_backtrack'if arm in ('R01','GR11')else'lite_fullstep'
def setup():
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='complete';BASE.mkdir(exist_ok=False)
    for name in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json'):(BASE/name).write_bytes((PRIOR/name).read_bytes())
    old=PRIOR/'trials/B';refs=read(old/'REFERENCE_LOCK.json');counts={ds:0 for ds in ('hc2','vidstg')};contracts=[]
    for box in ([.5,.5,.2,.4],[.1,.5,.2,.4],[.9,.1,.1,.1]):
        a=extent(box);assert a[:2]==box[:2]and a[2]>=box[2]-1e-15 and a[3]>=box[3]-1e-15;contracts.append(True)
    assert extent([.5,.5,.2,.4])==[.5,.5,.25,.5]
    for arm in ARMS:
        b=BASE/'trials'/arm;b.mkdir(parents=True);r=copy.deepcopy(refs)
        for rec in r['records']:
            orig=next(x for x in refs['records']if x['case_id']==rec['case_id'])
            for a,o in zip(rec['anchors'],orig['anchors']):
                if arm!='R01':a['box_cxcywh']=extent(o['box_cxcywh'])
                assert {k:v for k,v in a.items()if k!='box_cxcywh'}=={k:v for k,v in o.items()if k!='box_cxcywh'}
                assert a['box_cxcywh'][:2]==o['box_cxcywh'][:2]
            if arm=='R01':assert rec==orig
        r.update(reference_intervention='center-preserving capped extent x1.25'if arm!='R01'else'exact original B',original_B_reference_sha256=sha(old/'REFERENCE_LOCK.json'));put(b/'REFERENCE_LOCK.json',r);(b/'TEMPORAL_LOCK.json').write_bytes((old/'TEMPORAL_LOCK.json').read_bytes());d=copy.deepcopy(read(old/'DESIGN_LOCK.json'));d.update(trial_id=arm,scope='B extent x guarded reachability factorial, fixed sixteen sources',recipe=recipe(arm));put(b/'DESIGN_LOCK.json',d)
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=16,domain_counts={'hc2':8,'vidstg':8},baseline='exact prior B00',arms=list(ARMS),extent_factor=1.25,extent_center_fixed=True,step_scales={'full':[1.0],'backtrack':[1.0,.5,.25,.125]},gate='unchanged finite and equal-anchor mean Teacher IoU gain >= -1e-7; rollback and rejection early stop unchanged',no_GT_construction=True,no_new_teacher=True,all_exposed_development=True,bootstrap=False,CURRENT_change=False,official_fullset=False,GT_read=False))
    files=[Path(__file__),ROOT/'protocols/stvg_B_transfer_v1.md',BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'TEMPORAL_LOCK.json',PRIOR/'FINAL_COMPLETION.json',PRIOR/'trials/B/ROOT_READBACK.json',PRIOR/'trials/B/GLOBAL_PREDICTION_BARRIER.json',PRIOR/'trials/B/REFERENCE_LOCK.json',PRIOR/'trials/B/POSTSEAL_RESULTS.json',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_weightfree_coordinate_fit002.py',ROOT/'scripts/audit_stvg_spatial_lite_fit002.py',ROOT/'scripts/run_stvg_instance_student32_v1.py']
    put(BASE/'RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in files},GT_read=False))
    for arm in ARMS:
        b=BASE/'trials'/arm;rt=copy.deepcopy(read(old/'RUNTIME.json'));rt.update(trial_id=arm,scope='fixed B geometry and acceptance reachability')
        for p in [Path(__file__),BASE/'RUNTIME.json',BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json']:rt['pins'][str(p.relative_to(ROOT))]=sha(p)
        put(b/'RUNTIME.json',rt)
    put(BASE/'CPU_CONTRACTS.json',dict(status='pass',count=6,GT_read=False,CPU_only=True,extent_center_and_nonshrink_contracts=True,real_all_59_anchors_identity_position_weight_unchanged=True,R01_references_exact=True,recipes_original_unmodified=True));print('B_TRANSFER_SETUP_LOCKED')
def check(arm):
    for rel,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
    rt=read(BASE/'trials'/arm/'RUNTIME.json')
    for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
def execute(mode,arm):
    from scripts import run_stvg_instance_student32_v1 as old
    s=inspect.getsource(old.worker if mode=='worker'else old.root);tree=ast.parse(s)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(16),n)if type(n.value)is int and n.value==32 else n
    s=ast.unparse(ast.fix_missing_locations(Count().visit(tree)));assert s.count("'lite_fullstep'")==1;s=s.replace("'lite_fullstep'",repr(recipe(arm)))
    if mode=='root':assert s.count("bb = arr(o['boxes'])")==1;s=s.replace("bb = arr(o['boxes'])","bb = arr(o['boxes']).astype(float)")
    ns=dict(old.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',));exec(compile(s,str(Path(__file__)),'exec'),ns);ns[mode](arm)
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    from scripts.stvg_native_instance_common_v1 import write
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='B_transfer';g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for arm in ARMS:
            for mode in ('worker','root'):
                out=BASE/'execution'/arm/mode;out.mkdir(parents=True,exist_ok=False);write(BASE/'STATUS.json',dict(status='running',stage=mode,arm=arm,GT_read=False),mutable=True)
                with(out/'worker.log').open('xb')as log:
                    t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),mode,arm],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
                put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,(arm,mode)
        write(BASE/'STATUS.json',dict(status='measurements_complete_pending_independent_score_public_archive',stage='root_handoff',count=16,GT_read=False),mutable=True)
    finally:stop.set()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='setup':setup()
    elif mode=='supervise':supervise()
    else:execute(mode,sys.argv[2])
