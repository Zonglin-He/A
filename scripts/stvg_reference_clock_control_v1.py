"""One fixed-candidate Spatial WHO input-interface control, no prompt search.

Only fractional-time handling differs from the historical teacher worker.
Temporal-Lite outputs/observation centers are fixed, not silently recomputed.
"""
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,threading,time,traceback,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_geometry_audit_v1 import BASE as STUDY,LITE,read,put,sha,verify
BASE=STUDY/'clock_control'
UNIT='stvg-reference-clock-control-001.service'

def prepare():
    verify();roster=read(STUDY/'ROSTER_LOCK.json');ids={r['probe_alias']for r in roster['records']}
    source=LITE/'reclocked_inputs';lock=read(source/'TEACHER_JOB_LOCK.json');jobs=[copy.deepcopy(j)for j in lock['records']if j['case_id']in ids]
    assert len(jobs)==48 and len({j['job_id']for j in jobs})==48
    for job in jobs:job['time_encoding_version']='fractional_seconds_v2_spatial_only_fixed_Temporal_Lite_readout'
    put(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',records=jobs,count=48,GT_read=False))
    paths={v['path']:v['sha256']for j in jobs for v in j['videos']}
    for rel,h in paths.items():assert sha(ROOT/rel)==h
    put(BASE/'CANDIDATE_INPUT_BARRIER.json',dict(status='complete',count=12,teacher_job_count=48,files=[dict(path=p,sha256=h)for p,h in paths.items()],GT_read=False,source_job_lock_sha256=sha(source/'TEACHER_JOB_LOCK.json')))
    rt=copy.deepcopy(read(source/'TEACHER_RUNTIME.json'))
    rt.update(schema='stvg-reference-spatial-clock-interface-control-v1',scope='only model mRoPE fractional seconds dtype; exact original candidate video/prompt/FPS/weights/max output; fixed historical Temporal outputs and physical centers',status='locked',GT_read=False)
    for p in [Path(__file__),ROOT/'scripts/stvg_qwen_uniform_clock002.py',STUDY/'CLOCK_INTERFACE_AUDIT.json',STUDY/'ROSTER_LOCK.json',BASE/'TEACHER_JOB_LOCK.json',BASE/'CANDIDATE_INPUT_BARRIER.json']:
        rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(BASE/'TEACHER_RUNTIME.json',rt)
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=12,teacher_requests=48,change='one isolated fractional uniform-time dtype correction in Spatial WHO',fixed=['all historical candidate RGB and IDs','all physical observation centers','all prompts and generation settings','all historical Temporal-Lite output intervals','STVG-R1 weights','Student Source and final spatial parameters'],GT_read=False,legacy_responses_reused_as_corrected=False,geometry_variants_executed=False,
        stop_rule='no geometry patch without exact intermediary linkage and dominant correct-instance truncation; this interface control is not a full corrected-Temporal method or frame-selection gain',
        comparison='legacy vs corrected WHO IDs on identical saved videos; if references differ, real Student T+S decode under fixed parameters is required before any performance claim'))

def qualify(model,batch,job):
    import torch
    kw={k:batch[k]for k in ('input_ids','video_grid_thw','second_per_grid_ts','attention_mask')if k in batch}
    with torch.inference_mode():
        old,od=type(model.model).get_rope_index(model.model,**kw)
        new,nd=model.model.get_rope_index(**kw)
    h=lambda a:hashlib.sha256(a.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    assert model.config.vision_config.tokens_per_second==2
    return dict(status='pass',actual_prefill_positions_changed=not torch.equal(old,new),legacy_positions_sha256=h(old),corrected_positions_sha256=h(new),legacy_delta_sha256=h(od),corrected_delta_sha256=h(nd),grid=batch['video_grid_thw'].detach().cpu().tolist(),seconds_per_grid=list(map(float,batch['second_per_grid_ts'])),job_id=job['job_id'],GT_read=False)

def worker():
    from scripts.stvg_qwen_uniform_clock002 import apply
    source=ROOT/'scripts/run_stvg_evidence_teacher_v1.py';tree=ast.parse(source.read_text());original=ast.dump(copy.deepcopy(tree),include_attributes=False)
    class Patch(ast.NodeTransformer):
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if any(isinstance(t,ast.Name)and t.id=='m'for t in node.targets):
                return [node,ast.parse('_audit_clock_proof=_clock_apply(m)').body[0]]
            if any(isinstance(t,ast.Name)and t.id=='result'for t in node.targets):
                assert isinstance(node.value,ast.Call)
                node.value.keywords.extend([ast.keyword(arg='time_encoding_proof',value=ast.Name('_audit_clock_proof',ast.Load())),ast.keyword(arg='actual_time_encoding_readback',value=ast.Name('_audit_clock_actual',ast.Load()))])
            return node
        def visit_With(self,node):
            node=self.generic_visit(node)
            if any('inference_mode'in ast.unparse(x.context_expr)for x in node.items):return [ast.parse('_audit_clock_actual=_qualify(m,batch,job)').body[0],node]
            return node
    tree=ast.fix_missing_locations(Patch().visit(tree))
    class Strip(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name)and t.id.startswith('_audit_clock_')for t in node.targets):return None
            node=self.generic_visit(node)
            if any(isinstance(t,ast.Name)and t.id=='result'for t in node.targets):node.value.keywords=[k for k in node.value.keywords if k.arg not in ('time_encoding_proof','actual_time_encoding_readback')]
            return node
    assert ast.dump(Strip().visit(copy.deepcopy(tree)),include_attributes=False)==original
    scope=dict(__name__='scripts._reference_clock_control',__file__=str(source),_clock_apply=apply,_qualify=qualify)
    exec(compile(tree,str(source),'exec'),scope);scope['BASE']=BASE;scope['main']()

def supervise():
    from scripts import launch_stvg_instance_identity_v1 as resource
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='teacher';g=1024**3
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x)
    assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    state=resource.sample();assert state['available']>=8*g and state['gpu_free']>=26*g and state['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start();p=BASE/'execution';p.mkdir(exist_ok=False)
    put(p/'ADMISSION.json',dict(status='pass',properties=props,resources=state,GT_read=False,time=time.time()))
    try:
        with(p/'worker.log').open('xb')as f:
            t=time.perf_counter();child=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B',str(Path(__file__)),'worker'],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
            put(p/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=False,time=time.time()));code=child.wait()
        put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-t,GT_read=False,time=time.time()));assert code==0
    finally:stop.set()

if __name__=='__main__':
    try:{'prepare':prepare,'worker':worker,'supervise':supervise}[sys.argv[1]]()
    except BaseException as e:
        put(BASE/('FAILURE_'+sys.argv[1]+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
