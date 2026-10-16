"""Publish exact engineering provenance with each independently completed arm."""
from pathlib import Path
import ast,inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_controls256002 as previous
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.stvg_native_instance_common_v1 import read
def engineering_cost(kind,group):
    q=read(speed.BASE/'QUALIFICATION_COMPLETION.json')
    result=dict(status='complete',qualification_status=q['status'],engineering_adopted=q['engineering_adopted'],qualified_source_count=q['count'],new_GT_reads=0,scientific_parameters_acceptance_steps_and_precision_unchanged=True,all_original_step_audits_preserved=True,source_cache_boundary='exact pre-update Source views, before trainable LNs',final_prediction='all frames, both original decoder passes, original action gates and updated LNs recomputed; frozen prefix reused only after exact original full-forward qualification')
    result['qualification_comparisons']=[{k:r[k] for k in ('dataset','source_capture_seconds','prefix_restore_seconds','tests') if k in r} for r in q['records']]
    p=speed.BASE/'student_profiles'/(group+'.json')
    if p.exists():
        d=read(p);result['student']={k:v for k,v in d.items() if k not in ('rows',)};result['student_per_source']=d['rows']
    p=speed.BASE/'student_profiles'/(group+'_IO.json')
    if p.exists():result['checkpoint_IO']=read(p)
    t={'A_BT':'A','B_BT':'B'}.get(group,group);p=speed.BASE/'teacher_profiles'/(t+'.json')
    if p.exists():
        d=read(p);result['teacher_cache']=dict(count=d['count'],metadata_cache_new=sum(r['metadata_cache_new'] for r in d['records']),metadata_and_feature_prepare_seconds=sum(r['prepare_seconds'] for r in d['records']),extra_first8_comparisons=sum(r['qualification'] for r in d['records']),BF16_bits_losslessly_stored_as_uint16=True,precision_not_reduced=True,processor_metadata_reused=True,raw_job_ids_not_exported=True)
    result['separate_qualification_cost_is_shared_not_zero']=True
    return result
def source(use_A):
    text=previous.report_source(use_A)
    before="put(out/'CONFIG_AND_COST.json',anon(cost))";assert text.count(before)==1
    text=text.replace(before,"cost['engineering_speed_provenance']=engineering_cost(kind,group)\n    "+before)
    before="names=['report_stvg_controls256002.py',";assert text.count(before)==1
    names=['report_stvg_controls256_speed003.py',*__import__('scripts.prepare_stvg_pipeline_speed_v1',fromlist=['FILES']).FILES]
    text=text.replace(before,'names='+repr(names)[:-1]+",'report_stvg_controls256002.py',")
    before="runtime_sha256=sha(base/('NATIVE_DISPATCH_RUNTIME003.json' if kind=='core256' else 'EXECUTION_RUNTIME.json'))";assert text.count(before)==1
    text=text.replace(before,"runtime_sha256=sha(base/('NATIVE_DISPATCH_RUNTIME004.json' if kind=='core256' else 'EXECUTION_RUNTIME002.json'))")
    text=text.replace('The group completed full Student decoding,','The group completed exactness-qualified Student decoding over all frames (reusing only frozen pre-LN Source views where qualified),')
    text=text.replace('New full-model fits/readouts, teacher requests,','Actual new Source captures versus frozen-prefix restores, complete two-pass Decoder readouts, teacher requests,')
    before="doc=ROOT/'docs'/f'STVG_{kind.upper()}_{group}_001.md';";assert text.count(before)==1
    text=text.replace(before,"lines += ['', 'Engineering acceleration is qualified against the original complete Source/full-model path on fixed GT-free current-cohort parents in both domains. Initial predictions, trainable initialization, complete gradients/Adam/solver/acceptance paths and final boxes/logits/intervals must match exactly. Failed qualification retains original computation. Current research keeps every original step checkpoint, with bounded CPU asynchronous writing; no acceptance rule, solver iteration budget, update step budget or model precision is removed. Shared qualification and cache creation costs are reported separately and are not free.']\n    "+before)
    return text
def contracts():
    for a in (False,True):
        s=source(a);ast.parse(s);assert 'engineering_cost(kind,group)' in s and 'NATIVE_DISPATCH_RUNTIME004.json' in s
    print('SPEED_REPORT_ALL_SOURCE_BRANCHES_COMPILE_PASS_NO_GT_NO_MODEL')
def main(kind,group):
    use_A=False
    if kind=='core256':
        from scripts.add_stvg_controls256_A_comparisons_v1 import prepare
        use_A=prepare(group)
    ns=dict(previous.original.__dict__);ns.update(use_A=use_A,engineering_cost=engineering_cost)
    exec(compile(source(use_A),str(Path(__file__)),'exec'),ns);ns['main'](kind,group)
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else main(*sys.argv[1:3])
