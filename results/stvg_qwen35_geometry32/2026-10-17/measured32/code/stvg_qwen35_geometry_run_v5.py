"""All128 processor preflight; exact-uniform clocks are valid, original inputs unchanged."""
from pathlib import Path
import inspect,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_qwen35_geometry_run_v4 as prior,stvg_qwen35_geometry_run_v2 as old
BASE=old.BASE;read,write,sha=old.read,old.write,old.sha;parse_final=prior.parse_final
UNIT='stvg-qwen35-geometry32-005.service'
def native_video_adapter(*args,**kwargs):
    source=inspect.getsource(old.native_video_adapter).replace("assert fps>0 and context['encoder_uniform_spacing_approximation'] is True","assert fps>0 and type(context['encoder_uniform_spacing_approximation']) is bool").replace('historical_uniform_clock_approximation_retained=True',"historical_uniform_clock_approximation_retained=context['encoder_uniform_spacing_approximation']")
    ns=dict(old.__dict__);exec(compile(source,__file__,'exec'),ns);return ns['native_video_adapter'](*args,**kwargs)
def verify_run():
    prior.verify_run()
    for p,h in read(BASE/'EXECUTION_RUNTIME005.json')['pins'].items():assert sha(ROOT/p)==h,p
def init():
    import torch,transformers
    from scripts.stvg_qwen35_geometry_adapters_v2 import prepare_native_batch
    prior.verify_run();assert not(BASE/'EXECUTION_RUNTIME005.json').exists();processor=transformers.AutoProcessor.from_pretrained(old.CHECKPOINT,local_files_only=True,trust_remote_code=False);rows=[];missing=[]
    for j in read(BASE/'INPUT_JOB_LOCK.json')['records']:
        b,a=prepare_native_batch(processor,j,verified_video_adapter=native_video_adapter);rows.append(dict(job_id=j['job_id'],input_tensor_sha256=a['input_tensor_sha256'],exact_uniform=not j['context']['encoder_uniform_spacing_approximation'],GT_read=False));p=BASE/'formal_responses'/(j['job_id']+'.json')
        if p.exists():
            oldaudit=read(p)['input_audit'];assert a['input_tensor_sha256']==oldaudit['input_tensor_sha256']
        elif(BASE/'formal_claims'/(j['job_id']+'.json')).exists():missing.append(j['job_id'])
    assert len(rows)==128 and missing==['R007_0'];assert not torch.cuda.is_initialized()
    # This sole old claim stopped at prepare_native_batch, before model.generate.
    log=(BASE/'execution004/teacher/worker.log').read_text();assert 'native_video_adapter'in log and "context['encoder_uniform_spacing_approximation'] is True"in log
    saved=list((BASE/'formal_responses').glob('*.json'));assert len(saved)==26
    write(BASE/'recovery/exact_uniform_clock005/CPU_READBACK.json',dict(status='pass',count=128,records=rows,previous_responses=26,remaining_unique_requests=102,claimed_before_model_failure=missing,existing_26_actual_input_tensor_hashes_exact=True,exact_uniform_inputs=12,old_raw_outputs_preserved=True,Prompt_or_pixels_or_fps_or_coordinates_changed=False,new_model_calls=0,original_deadline_not_reset=True,GT_read=False))
    paths=[Path(__file__),ROOT/'scripts/continue_stvg_qwen35_geometry32_005.py',ROOT/'scripts/report_stvg_qwen35_geometry32_v5.py',BASE/'EXECUTION_RUNTIME004.json',BASE/'recovery/exact_uniform_clock005/CPU_READBACK.json'];write(BASE/'EXECUTION_RUNTIME005.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False))
def claim(job_id):
    from scripts.stvg_qwen35_geometry_adapters_v2 import claim_formal_request
    previous=BASE/'formal_claims'/(job_id+'.json')
    if previous.exists():assert job_id in read(BASE/'recovery/exact_uniform_clock005/CPU_READBACK.json')['claimed_before_model_failure'] and not(BASE/'formal_responses'/(job_id+'.json')).exists()
    else:claim_formal_request(job_id)
    write(BASE/'generation_claims005'/(job_id+'.json'),dict(status='claimed_once_before_generation',previous_preprocessor_claim_preserved=previous.exists(),GT_read=False))
def request(model,processor,job,qualification=False):
    p=BASE/'formal_responses'/(job['job_id']+'.json')
    if p.exists():return prior.request(model,processor,job,qualification)
    assert not qualification,'all qualification probes already completed'
    source=inspect.getsource(old.request).replace("else:claim_formal_request(job['job_id'])","else:claim(job['job_id'])")
    ns=dict(old.__dict__,verify_run=verify_run,parse_final=parse_final,native_video_adapter=native_video_adapter,claim=claim);exec(compile(source,__file__,'exec'),ns);return ns['request'](model,processor,job,qualification=False)
def teacher():
    def snapshot():verify_run();assert old.download_status()['snapshot_for_model_ready']
    def put(p,d,*args,**kwargs):
        if p==BASE/'QUALIFICATION_COMPLETION.json':
            d=dict(d,original_qualification_sha256=sha(p),all128_CPU_preflight_sha256=sha(BASE/'recovery/exact_uniform_clock005/CPU_READBACK.json'));p=BASE/'QUALIFICATION_COMPLETION005.json'
        write(p,d,*args,**kwargs)
    source=inspect.getsource(old.teacher).replace('stvg-qwen35-geometry32-002.service',UNIT).replace('ACTUAL_LIVE_RESOURCE_LIMITS.json','ACTUAL_LIVE_RESOURCE_LIMITS005.json').replace('actual_new_generations=128','actual_new_generations=102,reused_original_generations=26,total_actual_generations=128')
    ns=dict(old.__dict__,verify_run=verify_run,model_snapshot=snapshot,request=request,write=put);exec(compile(source,__file__,'exec'),ns);ns['teacher']()
def main(stage):
    verify_run()
    if stage=='teacher':return teacher()
    if stage=='reference_root':return prior.reference_root()
    return old.main(stage)
if __name__=='__main__':
    if sys.argv[1]=='init':init()
    else:main(sys.argv[1])
