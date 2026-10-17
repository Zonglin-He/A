"""Preserve exporter001; include actual launcher recovery and group costs."""
from pathlib import Path
import inspect,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_caption_coverage_v1 as original
from scripts.run_stvg_coverage224_v1 import BASE,EXP,read,put,sha
def main():
    source=inspect.getsource(original.main)
    needle="names=['analyze_stvg_caption_coverage256_v1.py'"
    assert source.count(needle)==1
    source=source.replace(needle,"names=['run_stvg_coverage224002.py','continue_stvg_coverage224002.py','report_stvg_caption_coverage002.py','publish_stvg_caption_coverage_v1.py','analyze_stvg_caption_coverage256_v1.py'")
    needle="    lines+=['','Public exports exclude"
    assert source.count(needle)==1
    source=source.replace(needle,"    coverage_cost_save(out,save)\n"+needle)
    def costs(out,save):
        assert read(BASE/'CPU_COMPLETION.json')['count']==256
        exits=[]
        for p in sorted((EXP/'execution').glob('*/WORKER_EXIT.json')):
            r=read(p);assert r['exit_code']==0;exits.append(dict(stage=p.parent.name,wall_seconds=r['wall_seconds']))
        teacher=read(EXP/'teacher/G32/TEACHER_GLOBAL_BARRIER.json');student=read(EXP/'POSTSEAL_RESULTS.json')['usage'];capture=read(BASE/'recovery/launcher_deadline002/CAPTURE.json')
        save('ACTUAL_EXECUTION_COST.json',dict(status='complete',count=256,new_development_parents=224,reused_initial_parents=32,stage_costs=exits,teacher={k:v for k,v in teacher.items() if isinstance(v,(int,float,bool)) or k=='status'},Student_fits=sum(r['actual_fit'] for r in student),accepted_steps=sum(r['accepted_steps'] for r in student),nonzero_final_sources=sum(r['nonzero_final'] for r in student),initial32_and_all_matched_controls_reused_not_zero_cost=True,launcher_recovery='002 adds missing finite-deadline proxy only; six completed short qualification calls retained; zero formal WHO/fit in failed attempt',GT_read=False))
    ns=dict(original.__dict__);ns['coverage_cost_save']=costs;exec(compile(source,__file__,'exec'),ns);ns['main']('coverage256')
if __name__=='__main__':main()
