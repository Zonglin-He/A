"""Publication adapter: same-cohort A comparisons, no duplicate inference/GT."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_controls256_v1 as original
def report_source(use_A):
    src=inspect.getsource(original.main)
    if use_A:
        before="cpu=read(b/'CPU_COMPLETION002.json');p=b/'POSTSEAL_RESULTS002.json'"
        after="cpu=read(b/'CPU_MATCHED_A003.json');p=b/'POSTSEAL_RESULTS_MATCHED_A003.json'"
        assert src.count(before)==1;src=src.replace(before,after)
        needle="assert cpu['status']=='complete' and cpu['count']==count and cpu['GT_read'] and cpu['result_sha256']==sha(p)"
        assert src.count(needle)==2;src=src.replace(needle,needle.replace("and cpu['GT_read'] and","and cpu['GT_read'] is False and"),1)
    needle="names=['report_stvg_controls256_v1.py',"
    assert src.count(needle)==1;src=src.replace(needle,"names=['report_stvg_controls256002.py','add_stvg_controls256_A_comparisons_v1.py','report_stvg_controls256_v1.py',")
    needle="contrasts=[group+'_minus_Temporal',group+'_minus_Frozen',group+'_minus_'+('A' if kind=='global32' else 'Full_E1')]"
    assert src.count(needle)==1
    src=src.replace(needle,needle+"\n    if kind=='core256' and use_A and group!='A':contrasts=list(dict.fromkeys(contrasts+[group+'_minus_A']))")
    needle="doc=ROOT/'docs'/f'STVG_{kind.upper()}_{group}_001.md';"
    assert src.count(needle)==1
    notice="\n    if kind=='core256':lines += ['',('Matched A uses all the same256 parents and common E1 evaluation support, copied from sealed A scalar results with shared Frozen/Temporal/Full_E1 values independently exact-checked; no new model or GT reader. Arm-specific adopted-support values are not merged.' if use_A else 'Matched A is pending because the A256 group has not completed. This first Full_E1 group is published immediately with Frozen/Temporal controls; the complete A group will supply their same-cohort paired comparison. Historical16-source A is not substituted.') ]\n    "
    src=src.replace(needle,notice+needle)
    return src
def contracts():
    import ast
    for use_A in (False,True):
        src=report_source(use_A);ast.parse(src);compile(src,str(Path(__file__)),'exec')
        assert ('CPU_MATCHED_A003.json' in src)==use_A
    print('REPORT002_GENERATED_BRANCHES_COMPILE_PASS_NO_GT_NO_MODEL')
def main(kind,group):
    use_A=False
    if kind=='core256':
        from scripts.add_stvg_controls256_A_comparisons_v1 import prepare
        use_A=prepare(group)
    ns=dict(original.__dict__);ns.update(use_A=use_A);exec(compile(report_source(use_A),str(Path(__file__)),'exec'),ns);ns['main'](kind,group)
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else main(*sys.argv[1:3])
