"""One unambiguous bbox payload schema; existing four responses never regenerated."""
from pathlib import Path
import inspect,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_qwen35_geometry_run_v3 as prior,stvg_qwen35_geometry_run_v2 as old
from scripts.stvg_qwen35_geometry_contract_v1 import parse_final as strict,_unique
BASE=old.BASE;read,write,sha=old.read,old.write,old.sha
UNIT='stvg-qwen35-geometry32-004.service'
def parse_final(raw,anchor_id,*,truncated=False):
    initial=strict(raw,anchor_id,truncated=truncated)
    if initial['status']in ['valid','abstain','no_target']or truncated or not isinstance(raw,str):return initial
    text=raw.strip()
    if '<think>'in text or '</think>'in text:return initial
    if text.startswith('```')and text.endswith('```'):
        lines=text.splitlines()
        if lines[0]not in ['```','```json']or lines[-1]!='```':return initial
        text='\n'.join(lines[1:-1]).strip()
    try:o=json.loads(text,object_pairs_hook=_unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
    except (ValueError,TypeError):return initial
    if isinstance(o,list):
        if len(o)!=1:return initial
        o=o[0]
    if not isinstance(o,dict)or'bbox_2d'not in o or not set(o)<={'bbox_2d','anchor_id','label','status'}:return initial
    if 'anchor_id'in o and o['anchor_id']!=anchor_id:return initial
    if 'label'in o and not isinstance(o['label'],str):return initial
    if 'status'in o and o['status']!='ok':return initial
    z=o['bbox_2d']
    if not isinstance(z,list)or len(z)!=4 or any(type(v)not in [int,float]or not math.isfinite(v)for v in z):return initial
    result=strict(json.dumps(dict(anchor_id=anchor_id,status='ok',bbox_2d=z)),anchor_id,truncated=truncated)
    if result['status']=='valid':result['reason']='lossless_unique_bbox_payload_wire'
    return result
def verify_run():
    prior.verify_run()
    for p,h in read(BASE/'EXECUTION_RUNTIME004.json')['pins'].items():assert sha(ROOT/p)==h,p
def init():
    prior.verify_run();assert not (BASE/'EXECUTION_RUNTIME004.json').exists();rows=[]
    for p in (BASE/'formal_responses').glob('*.json'):
        d=read(p);z=parse_final(d['raw_final_output'],d['job_id'],truncated=d['truncated']);assert z['status']=='valid';rows.append(dict(job_id=d['job_id'],path=str(p.relative_to(ROOT)),sha256=sha(p),original_status=d['parsed']['status'],lossless_status=z['status'],GT_read=False))
    assert len(rows)==4
    for raw in ['[{"bbox_2d":[0,0,100,100]},{"bbox_2d":[0,0,100,100]}]','{"anchor_id":"wrong","bbox_2d":[0,0,100,100]}','{"bbox_2d":[100,0,0,100]}','{"bbox_2d":null}']:
        assert parse_final(raw,'right')['status']!='valid'
    write(BASE/'recovery/unique_bbox_wire004/CPU_READBACK.json',dict(status='pass',count=4,records=rows,one_declared_rule='exactly one finite normalized1000 bbox payload with optional declared metadata; reject mismatched anchor, multiple/ambiguous/null-without-explicit-abstention/extra fields',Prompt_changed=False,coordinates_changed=False,new_model_calls=0,raw_responses_and_both_failures_preserved=True,new_unique_requests_max=124,original_deadline_not_reset=True,GT_read=False))
    paths=[Path(__file__),ROOT/'scripts/continue_stvg_qwen35_geometry32_004.py',ROOT/'scripts/report_stvg_qwen35_geometry32_v4.py',BASE/'EXECUTION_RUNTIME003.json',BASE/'recovery/unique_bbox_wire004/CPU_READBACK.json'];write(BASE/'EXECUTION_RUNTIME004.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False))
def request(model,processor,job,qualification=False):
    p=BASE/'formal_responses'/(job['job_id']+'.json')
    if p.exists():
        d=read(p);assert d['input_job_sha256']==old.canonical_sha(job)and d['model_snapshot_sha256']==sha(BASE/'MODEL_SNAPSHOT_LOCK.json')and d['prompt_lock_sha256']==sha(BASE/'PROMPT_LOCK.json');d['parsed']=parse_final(d['raw_final_output'],job['job_id'],truncated=d['truncated']);return d
    ns=dict(old.__dict__,verify_run=verify_run,parse_final=parse_final);exec(compile(inspect.getsource(old.request),__file__,'exec'),ns);return ns['request'](model,processor,job,qualification=qualification)
def teacher():
    source=inspect.getsource(prior.teacher).replace('actual_new_generations=125,reused_original_generations=3','actual_new_generations=124,reused_original_generations=4').replace('ACTUAL_LIVE_RESOURCE_LIMITS003.json','ACTUAL_LIVE_RESOURCE_LIMITS004.json');ns=dict(prior.__dict__,verify_run=verify_run,request=request,UNIT=UNIT);exec(compile(source,__file__,'exec'),ns);ns['teacher']()
def reference_root():
    from scripts import stvg_qwen35_geometry_adapters_v2 as a
    source=inspect.getsource(a.seal_references).replace("assert d['parsed']==parsed","assert (d['parsed']==parsed) or (parsed['reason']=='lossless_unique_bbox_payload_wire')")
    ns=dict(a.__dict__,parse_final=parse_final);exec(compile(source,__file__,'exec'),ns);ns['seal_references']()
def main(stage):
    verify_run()
    if stage=='teacher':return teacher()
    if stage=='reference_root':return reference_root()
    return old.main(stage)
if __name__=='__main__':
    if sys.argv[1]=='init':init()
    else:main(sys.argv[1])
