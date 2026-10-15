"""Pin a complete no-GT confirmation handoff after frozen development selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_tuning_v1';E=B/'confirmation003'
def main():
 assert not (E/'GT_EXPOSURE.json').exists()
 sel=read(B/'METHOD_SELECTION_LOCK.json');assert sel['status']=='locked'and not sel['confirmation_GT_read']
 base=read(B/'STUDENT_RUNTIME.json');pins=dict(base['pins']);pins.update(read(E/'RUNTIME.json')['pins'])
 paths=[B/'METHOD_SELECTION_LOCK.json',B/'CONFIRMATION_ROSTER_LOCK.json',E/'REFERENCE_LOCK.json',E/'TEMPORAL_GLOBAL_BARRIER.json',E/'TEACHER_GLOBAL_BARRIER.json',E/'ROOT_INPUT_RENDER_READBACK.json',E/'ROOT_COHORT_READBACK.json',E/'CPU_SELECTED_CONFIG_CONTRACTS.json']
 paths +=[ROOT/'scripts'/n for n in ['run_stvg_weightfree_expand_student_v1.py','stvg_weightfree_selected_fit_v1.py','review_stvg_weightfree_expand_v1.py','score_stvg_weightfree_expand_v1.py','lock_stvg_weightfree_confirmation_student_v1.py']]
 paths +=[ROOT/'protocols/stvg_weightfree_parameter_addendum002.md']
 for f in read(E/'TEMPORAL_GLOBAL_BARRIER.json')['files']:pins[f['path']]=f['sha256']
 for p in paths:pins[str(p.relative_to(ROOT))]=sha(p)
 for p,h in pins.items():assert sha(ROOT/p)==h,p
 write(E/'STUDENT_RUNTIME.json',dict(status='locked',pins=pins,GT_read=False,confirmation_GT_read=False,development_selection_used_labels=True))
 print('locked',len(pins))
if __name__=='__main__':main()
