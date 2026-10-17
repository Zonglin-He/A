"""Completed root-cause/CPU repair stage, explicitly before new Caption GPU work."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import BASE,read,put,sha,verify
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy
def main():
    verify();d=read(BASE/'DIAGNOSIS.json');origin=read(BASE/'ROOT_MISSING_ORIGIN_COMPLETION.json');contracts=read(BASE/'CPU_CONTRACT_COMPLETION.json');inputs=read(BASE/'INPUT_BARRIER.json');assert origin['count']==contracts['count']==inputs['count']==54
    out=ROOT/'results/stvg_spatial_caption_repair_v1/2026-10-17/diagnosis001';out.mkdir(parents=True,exist_ok=False)
    put(out/'DIAGNOSIS_SUMMARY.json',dict(status='complete_CPU_diagnosis_and_implementation',count=384,missing=54,cohorts=d['summary'],root_cause=d['root_cause'],new_Caption_generation_status='pending original live feature worker normal exit; not claimed complete',new_GT_reader_calls=0,new_model_calls_in_this_report=0,GT_read=False))
    for name in ('ROOT_MISSING_ORIGIN_COMPLETION.json','CPU_CONTRACT_COMPLETION.json','REPAIR_PLAN_LOCK.json'):put(out/name,read(BASE/name))
    doc=ROOT/'docs/STVG_CAPTION_MISSING_DIAGNOSIS001.md';doc.write_text('''# Caption missingness: completed diagnosis and CPU-validated repair

Core256 is missing35 Captions (HC8/Vid27); confirmation128 is missing19 (HC3/Vid16). All54 queries are absent from both the old Temporal input-job list and its actual Teacher barrier. These are pre-generation skips, not executed R1 empty output or a written file disappearing.53 skipped because deduplication left fewer than four distinct temporal candidates; the other query failed a non-Native expanded candidate sample binding. Every affected Native candidate independently has the original eight valid Source-bound frames.

The bug is coupling independent Spatial Caption acquisition to permission for the entire four-candidate Temporal update. The new per-candidate API verifies the requested existing clip's interval, canonical clock and exact Source sampling binding. Original Temporal admission and fallback remain unchanged. It does not invent a fourth candidate, duplicate frames, change interval radii, relax acceptance, or modify old results.111 CPU contract checks pass across the54 real failure cases, including unchanged rejection by the original Temporal gate and negative controls for short clips and bad sample bindings.54 real marked8x448 inputs pass original rendering and Source/canonical RGB hash checks.

Repair implementation is deployed in a new version. The live GPU feature worker is allowed to finish normally; only its CPU coordinator is paused, with an independent replacement of the same resource guard. No live GPU worker is signalled and no second model is loaded. Then54 actual Captions and two-domain exact original-generator qualification are scheduled before the new Native A/B evidence stages. Caption generation is still pending at this publication: CPU input checks do not prove successful generation or Student improvement. The original8h Native window is not reset.

New Caption bindings are versioned; original locks, noCoverage results and failed confirmation remain immutable. Confirmation128 is not rescored as a fresh confirmation. Same-clock Native/Adapted controls reuse the exact original E1 inputs; remaining Native observations retain their prepared inputs. Corrected Full-B may reuse a whole executed Native-B prediction on the35 affected fallback queries only after exact Source, complete E1 jobs, final time, recipe and seed checks. No per-frame/window/state splicing or GT-based routing is allowed. Old Full-B remains separately reported.

Teacher weights, generic Caption Prompt, greedy160-token budget,8x448 inputs, clocks, visual budget, Source/Temporal/spatial parameters and rejection rules stay fixed. Full actual Caption and corrected Student effects require their own future completion receipts and publication. No CURRENT change, official fullset, Prompt/LR/token search or new module.

Private query/Caption/response text, media, annotations/GT coordinates, raw tensors/features, weights, optimizer states, credentials and conversation records are excluded.
''')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([-.17,.83],[8,3],.34,label='HC');ax.bar([.17,1.17],[27,16],.34,label='VidSTG');ax.set(xticks=[0,1],xticklabels=['core256','confirmation128'],ylabel='Caption never requested: parent sources');ax.legend();fig.tight_layout();fig.savefig(out/'missing_origins.png',dpi=150);plt.close(fig)
    for name in ('stvg_spatial_caption_repair_v1.py','continue_stvg_captionfix_native002.py','score_stvg_factorial_captionfix002.py','report_stvg_caption_diagnosis001.py','publish_stvg_caption_diagnosis001.py','report_stvg_caption_repair_v1.py','report_stvg_factorial_captionfix002.py'):
        p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    paths={str(doc.relative_to(ROOT))}|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths);put(BASE/'publication/diagnosis001/EXPORT_MANIFEST.json',dict(status='complete',count=384,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('CAPTION_DIAGNOSIS_IMPLEMENTATION_EXPORTED_PENDING_ACTUAL_PIXELS_REMOTE')
if __name__=='__main__':main()
