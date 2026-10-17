"""Matched448 report; low-resolution pilot and its complete cost retained."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_qwen_b3_temporal32_v1 as old
from scripts.stvg_qwen_b3_temporal32_448_v2 import BASE,read,write,sha
OUT=ROOT/'results/stvg_qwen_temporal/2026-10-17/b3_matched32_448'
def report():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    old.BASE=BASE;old.OUT=OUT;old.report();cost=read(OUT/'COST.json');cost['prior_low_resolution_stages']=[dict(stage=p.parent.name,**read(p))for p in(BASE.parent/'execution').glob('*/WORKER_EXIT.json')];cost.update(prior_teacher_calls=read(BASE.parent/'TEACHER_BARRIER.json')['calls'],corrected_teacher_calls=read(BASE/'TEACHER_BARRIER.json')['calls'],budget_not_reset=True);write(OUT/'COST.json',cost,mutable=True)
    write(OUT/'PROCESSOR_CORRECTION.json',dict(status='pass',count=32,only_Qwen_total_video_pixel_budget_corrected=True,original_eight448_NPY_bytes_times_and_Prompt_unchanged=True,old64_tokens_per_clip_not_a_matched_resolution_comparison=True,new784_tokens_per_clip_verified=True,prior_raw_outputs_fits_predictions_and_failure_pilot_preserved=True,correction_selected_before_reading_pilot_scores=True,original_deadline_not_reset=True,GT_read=False))
    p=OUT/'REPORT.md';p.write_text(p.read_text()+'\n\nProcessor correction002: Qwen3-VL interprets longest_edge as total video pixels. Copying the R1 per-frame448^2 limit had resized8frames to128 each (64 vision tokens/clip). That pilot remains preserved and its calls/cost included. Correcting total budget to8*448^2 preserves8x448, actual native grid[4,28,28]/784vision tokens verified before and during every model request. Hypotheses/NPY/clock/Prompt/reward/Student unchanged; this is an input contract correction, not an outcome retry or resolution sweep. Original4h deadline inherited. Executable entry is stvg_qwen_b3_temporal32_448_v2.py.\n')
    for name in ['stvg_qwen_b3_temporal32_448_v2.py','continue_stvg_qwen_b3_temporal32_448_v2.py','report_stvg_qwen_b3_temporal32_448_v2.py']:(OUT/'code'/name).write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    m=BASE/'publication/temporal32/EXPORT_MANIFEST.json';d=read(m);paths=set(d['paths'])|{str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()};verify_privacy(paths);d.update(paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]);write(m,d,mutable=True)
if __name__=='__main__':report()
