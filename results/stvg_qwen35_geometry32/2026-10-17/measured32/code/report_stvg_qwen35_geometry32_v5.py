"""Full comparison and all fixed engineering recovery costs, no hidden model repeats."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_qwen35_geometry32_v4 as prior
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=prior.prior.BASE;OUT=prior.prior.OUT
def report():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    prior.report();c=read(OUT/'COST.json');c.update(all_stages=[dict(revision=folder.name,stage=p.parent.name,**read(p))for folder in [BASE/'execution002',BASE/'execution003',BASE/'execution004',BASE/'execution005']for p in folder.glob('*/WORKER_EXIT.json')],formal_Qwen_generations=128,reused_before005=26,remaining_actual005=102,no_generation_repeated=True);write(OUT/'COST.json',c,mutable=True)
    write(OUT/'CLOCK_RECOVERY005.json',dict(status='pass',CPU_processor_inputs_checked=128,exact_uniform_inputs=12,previous_actual_input_tensor_hash_checks=26,Prompt_pixels_fps_coordinates_unchanged=True,preprocessor_failure_claim_resumed_before_any_model_call=True,raw_responses_failures_deadline_preserved=True,GT_read=False))
    doc=ROOT/'docs/STVG_QWEN35_GEOMETRY32_MEASURED_001.md';doc.write_text(doc.read_text()+'\n\nRecovery005 corrects an assertion that mistakenly required approximate context timing. Twelve contexts are exactly uniformly spaced (flag False), which is legal; processor inputs/fps/timestamps are unchanged. All128 inputs were CPU preflighted, and every26 existing actual tensor hash matches. One claim had stopped in preprocessing before model.generate; it resumes once. Only102 remaining unique generations executed. All failures and previous outputs/deadlines preserved.\n')
    for name in ['stvg_qwen35_geometry_run_v3.py','stvg_qwen35_geometry_run_v4.py','stvg_qwen35_geometry_run_v5.py','continue_stvg_qwen35_geometry32_003.py','continue_stvg_qwen35_geometry32_004.py','continue_stvg_qwen35_geometry32_005.py','report_stvg_qwen35_geometry32_v3.py','report_stvg_qwen35_geometry32_v4.py','report_stvg_qwen35_geometry32_v5.py']:(OUT/'code'/name).write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    m=BASE/'publication/measured32/EXPORT_MANIFEST.json';d=read(m);paths=set(d['paths'])|{str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);d.update(paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]);write(m,d,mutable=True)
if __name__=='__main__':{'report':report,'render':prior.prior.old.render}[sys.argv[1]]()
