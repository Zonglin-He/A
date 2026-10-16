"""Include the shared qualification wall cost and its actual additional work."""
from pathlib import Path
import ast,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_controls256_speed003 as previous
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.stvg_native_instance_common_v1 import read
def engineering_cost(kind,group):
    d=previous.engineering_cost(kind,group);q=read(speed.BASE/'QUALIFICATION_COMPLETION.json');passed=q['status']=='pass'
    comparisons=sum(len(r['tests']) for r in q['records'])
    d['shared_qualification_wall_seconds']=q['wall_seconds'];d['qualification_work_scope']='Shared one-time comparisons, not a new scientific arm or free preprocessing. Failure costs remain charged.'
    d['qualification_actual_work']=dict(source_captures=len(q['records']) if passed else None,source_prefix_restores=len(q['records']) if passed else None,original_plus_cached_fit_executions=2*comparisons if passed else None,original_RGB_final_forwards=comparisons if passed else None,complete_prefix_final_readouts=comparisons if passed else None,counts_complete=passed)
    if 'teacher_cache' in d:d['teacher_cache']['additional_short8_generation_calls']=2*d['teacher_cache']['extra_first8_comparisons']
    return d
def source(use_A):
    s=previous.source(use_A);needle="names=['report_stvg_controls256_speed003.py',";assert s.count(needle)==1
    return s.replace(needle,"names=['report_stvg_controls256_speed004.py','profile_stvg_pipeline_saved_v1.py','report_stvg_controls256_speed003.py',")
def contracts():
    for x in (False,True):ast.parse(source(x))
    print('SPEED004_PUBLIC_COST_BRANCHES_COMPILE_PASS_NO_MODEL_NO_GT')
def main(kind,group):
    use_A=False
    if kind=='core256':
        from scripts.add_stvg_controls256_A_comparisons_v1 import prepare
        use_A=prepare(group)
    ns=dict(previous.previous.original.__dict__);ns.update(use_A=use_A,engineering_cost=engineering_cost)
    exec(compile(source(use_A),str(Path(__file__)),'exec'),ns);ns['main'](kind,group)
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else main(*sys.argv[1:3])
