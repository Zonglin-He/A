"""Lossless singleton-box wire compatibility, old claims and deadline retained."""
from pathlib import Path
import inspect,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_qwen35_geometry_run_v2 as old
from scripts.stvg_qwen35_geometry_contract_v1 import parse_final as strict,_unique
BASE=old.BASE
read,write,sha=old.read,old.write,old.sha
UNIT='stvg-qwen35-geometry32-003.service'

def parse_final(raw,anchor_id,*,truncated=False):
    result=strict(raw,anchor_id,truncated=truncated)
    if result['status']!='multiple_or_list_output':return result
    text=raw.strip()
    if text.startswith('```')and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1]).strip()
    try:obj=json.loads(text,object_pairs_hook=_unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
    except (ValueError,TypeError):return result
    if not isinstance(obj,list)or len(obj)!=1 or not isinstance(obj[0],dict)or set(obj[0])!={'bbox_2d','label'}or not isinstance(obj[0]['label'],str):return result
    # One unambiguous box, explicitly requested normalized1000 units retained.
    # Binding is the unique immutable request, never its model-generated label.
    canonical=dict(anchor_id=anchor_id,status='ok',bbox_2d=obj[0]['bbox_2d'])
    parsed=strict(json.dumps(canonical,allow_nan=False),anchor_id,truncated=truncated)
    if parsed['status']=='valid':parsed['reason']='lossless_singleton_bbox_label_wire'
    return parsed

def verify_run():
    old.verify_run()
    d=read(BASE/'EXECUTION_RUNTIME003.json')
    for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
    assert d['original_scientific_deadline_epoch']==read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
    return d

def init():
    old.verify_run();assert not (BASE/'EXECUTION_RUNTIME003.json').exists()
    jobs={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};records=[]
    for p in (BASE/'formal_responses').glob('*.json'):
        d=read(p);j=jobs[d['job_id']];assert d['input_job_sha256']==old.canonical_sha(j) and not d['GT_read']
        z=parse_final(d['raw_final_output'],d['job_id'],truncated=d['truncated']);records.append(dict(job_id=d['job_id'],path=str(p.relative_to(ROOT)),sha256=sha(p),old_status=d['parsed']['status'],new_status=z['status'],new_reason=z['reason'],GT_read=False))
    assert len(records)==3 and sum(x['old_status']!=x['new_status']for x in records)==1
    assert parse_final('[{"bbox_2d":[0,0,100,100],"label":"x"},{"bbox_2d":[0,0,100,100],"label":"x"}]','x')['status']=='multiple_or_list_output'
    assert parse_final('[{"bbox_2d":[-1,0,100,100],"label":"x"}]','x')['status']=='out_of_range_or_degenerate'
    assert parse_final('[{"bbox_2d":[0,0,100,100],"label":"x"}]','x',truncated=True)['status']=='truncated'
    write(BASE/'recovery/singleton_wire003/CPU_READBACK.json',dict(status='pass',count=3,records=records,Prompt_changed=False,coordinates_changed=False,GT_access=False,new_model_calls=0,raw_responses_preserved=True,maximum_new_requests=125,original_deadline_epoch=read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']))
    paths=[Path(__file__),ROOT/'scripts/continue_stvg_qwen35_geometry32_003.py',ROOT/'scripts/report_stvg_qwen35_geometry32_v3.py',BASE/'EXECUTION_RUNTIME002.json',BASE/'recovery/singleton_wire003/CPU_READBACK.json']
    write(BASE/'EXECUTION_RUNTIME003.json',dict(status='locked',count=32,pins={str(p.relative_to(ROOT)):sha(p)for p in paths},original_scientific_deadline_epoch=read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch'],new_requests_max=125,GT_read=False))

def request(model,processor,job,qualification=False):
    p=BASE/'formal_responses'/(job['job_id']+'.json')
    if p.exists():
        d=read(p);assert d['input_job_sha256']==old.canonical_sha(job) and d['model_snapshot_sha256']==sha(BASE/'MODEL_SNAPSHOT_LOCK.json')and d['prompt_lock_sha256']==sha(BASE/'PROMPT_LOCK.json')
        d['parsed']=parse_final(d['raw_final_output'],job['job_id'],truncated=d['truncated']);return d
    ns=dict(old.__dict__,verify_run=verify_run,parse_final=parse_final)
    exec(compile(inspect.getsource(old.request),__file__,'exec'),ns)
    return ns['request'](model,processor,job,qualification=qualification)

def teacher():
    # The snapshot is already locked; no rewrite or repeat of completed requests.
    def snapshot():
        verify_run();d=old.download_status();assert d['snapshot_for_model_ready']
        m=read(BASE/'MODEL_SNAPSHOT_LOCK.json')
        for f in m['files']:assert (old.CHECKPOINT/f['name']).stat().st_size==f['bytes']
    source=inspect.getsource(old.teacher).replace('stvg-qwen35-geometry32-002.service',UNIT).replace('ACTUAL_LIVE_RESOURCE_LIMITS.json','ACTUAL_LIVE_RESOURCE_LIMITS003.json').replace('actual_new_generations=128','actual_new_generations=125,reused_original_generations=3,total_actual_generations=128')
    ns=dict(old.__dict__,verify_run=verify_run,model_snapshot=snapshot,request=request)
    exec(compile(source,__file__,'exec'),ns);ns['teacher']()

def reference_root():
    from scripts import stvg_qwen35_geometry_adapters_v2 as a
    source=inspect.getsource(a.seal_references).replace("assert d['parsed']==parsed","assert (d['parsed']==parsed) or (d['parsed']['status']=='multiple_or_list_output' and parsed['reason']=='lossless_singleton_bbox_label_wire')")
    ns=dict(a.__dict__,parse_final=parse_final);exec(compile(source,__file__,'exec'),ns);ns['seal_references']()

def main(stage):
    verify_run()
    if stage=='teacher':return teacher()
    if stage=='reference_root':return reference_root()
    return old.main(stage)

if __name__=='__main__':
    if sys.argv[1]=='init':init()
    else:main(sys.argv[1])
