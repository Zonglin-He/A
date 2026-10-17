"""Extend the already locked R1 slot0 interface to32 parents, reusing eight exact outputs."""
from pathlib import Path
import ast,copy,inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts import stvg_semantic_preference_interface_v2 as old
BASE=ROOT/'artifacts/stvg_semantic_preference_r1_32_v1';PRIOR=old.BASE

def prepare():
    assert not BASE.exists();BASE.mkdir()
    for n in ['ROSTER_LOCK.json','INPUT_JOB_LOCK.json','INPUT_BARRIER.json']:(BASE/n).write_bytes((PRIOR/n).read_bytes())
    jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];probes=[j['job_id'] for j in jobs if j['slot']==0];assert len(probes)==32
    write(BASE/'QUALIFICATION_PLAN.json',dict(status='locked',count=32,probe_jobs=probes,maximum_total_R1_prefill_calls=64,maximum_new_R1_prefill_calls=48,new_generations=0,GT_read=False))
    rows=[]
    for f in read(PRIOR/'teachers/R1/QUALIFICATION_COMPLETION.json')['files']:
        original=ROOT/f['path'];assert sha(original)==f['sha256'];r=read(original);assert r['job_id'] in probes
        p=BASE/'teachers/R1'/original.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(original.read_bytes());rows.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    assert len(rows)==8
    write(BASE/'REUSE_LOCK.json',dict(status='locked',count=8,files=rows,original_input_job_sha256=sha(PRIOR/'INPUT_JOB_LOCK.json'),current_input_job_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),exact_same_input_prompt_candidate_model=True,prior_prefill_calls=16,new_model_calls=0,GT_read=False))
    write(BASE/'R1_PRIORITY_AUTHORIZATION002.json',dict(status='explicitly_authorized',user_instruction='用完整32来源计算语义偏好最终IoU',GT_read=False))

def collect():
    from scripts.stvg_semantic_preference_student32_v1 import verify
    reuse=read(BASE/'REUSE_LOCK.json');assert reuse['original_input_job_sha256']==reuse['current_input_job_sha256']
    for f in reuse['files']:assert sha(ROOT/f['path'])==f['sha256']
    source=inspect.getsource(old.collect)
    source=source.replace("probes=", "probes=")
    needle="probe=read(BASE/'QUALIFICATION_PLAN.json')['probe_jobs'];outputs=[];calls=0"
    assert source.count(needle)==1
    source=source.replace(needle,"probe=[j for j in read(BASE/'QUALIFICATION_PLAN.json')['probe_jobs'] if not (BASE/'teachers/R1'/(j+'.json')).exists()];outputs=list(reuse_rows);calls=0")
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(32),n) if type(n.value)is int and n.value==8 else n
    source=ast.unparse(ast.fix_missing_locations(Count().visit(ast.parse(source))))
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify,reuse_rows=reuse['files']);exec(compile(source,__file__,'exec'),ns);ns['collect']('R1')
    d=read(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json');assert d['count']==32 and d['actual_prefill_calls']==48 and len(d['files'])==32
    keys=set()
    for f in d['files']:
        assert sha(ROOT/f['path'])==f['sha256'];r=read(ROOT/f['path']);assert r['job_id'] not in keys;keys.add(r['job_id'])
        for e in r['evidence']:
            p=old.categorical_from_logprobs(e['raw_label_logprobs']);assert old.restore_candidate_order(p,e['display_order'])==e['candidate_order_posterior']
    assert keys==set(read(BASE/'QUALIFICATION_PLAN.json')['probe_jobs'])
    write(BASE/'R1_ROOT_READBACK.json',dict(status='pass',count=32,new_prefill_calls=48,reused_prefill_calls=16,total_actual_prefill_calls=64,all_prior_eight_queries_reused_exact=True,GT_read=False))
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else collect()
