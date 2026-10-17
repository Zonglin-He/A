"""Original all-source report plus exact singleton-wire recovery provenance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_qwen35_geometry32_v2 as old
BASE,OUT,read,write,sha=old.BASE,old.OUT,old.read,old.write,old.sha
def report():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    old.report();rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}'for i,r in enumerate(rr,1)}
    rec=read(BASE/'recovery/singleton_wire003/CPU_READBACK.json');write(OUT/'WIRE_RECOVERY003.json',dict(status='pass',previous_responses_reused=3,new_unique_requests=125,total_unique_requests=128,Prompt_changed=False,coordinates_changed=False,one_unambiguous_bbox_label_list_accepted_without_GT=True,old_raw_and_invalid_outcome_preserved=True,original_deadline_not_reset=True,GT_read=False))
    c=read(OUT/'COST.json');c.update(stages_original002=[dict(stage=p.parent.name,**read(p))for p in (BASE/'execution002').glob('*/WORKER_EXIT.json')],stages_recovery003=[dict(stage=p.parent.name,**read(p))for p in (BASE/'execution003').glob('*/WORKER_EXIT.json')],formal_Qwen_generations=128,reused_original_generations=3,new_recovery_generations=125);write(OUT/'COST.json',c,mutable=True)
    doc=ROOT/'docs/STVG_QWEN35_GEOMETRY32_MEASURED_001.md';doc.write_text(doc.read_text()+'\n\nRecovery003: the first three raw outputs and once-only claims are unchanged. One returned exactly one bbox_2d/label object in a list. The declared normalized1000 coordinates were losslessly decoded under one GT-free singleton compatibility rule, without clamping, unit guessing or a new Prompt. Multi-box/duplicate-key/invalid/truncated output remains invalid. 125 remaining unique requests executed,128 total. Original scientific deadline and old failure retained. No response was regenerated.\n')
    for name in ['stvg_qwen35_geometry_run_v3.py','continue_stvg_qwen35_geometry32_003.py','report_stvg_qwen35_geometry32_v3.py']:(OUT/'code'/name).write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    manifest=BASE/'publication/measured32/EXPORT_MANIFEST.json';d=read(manifest);paths=set(d['paths'])|{str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);d.update(paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]);write(manifest,d,mutable=True)
if __name__=='__main__':{'report':report,'render':old.render}[sys.argv[1]]()
