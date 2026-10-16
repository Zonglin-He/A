"""Export the sealed matched fractional-clock Temporal32 group."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def main():
    out=ROOT/'results/stvg_instance_evidence32/2026-10-16/temporal001';assert not out.exists();out.mkdir(parents=True)
    b=BASE/'temporal';d=read(b/'POSTSEAL_RESULTS.json');cpu=read(b/'CPU_COMPLETION.json');r=read(b/'ROOT_READBACK.json');t=read(b/'TEACHER_GLOBAL_BARRIER.json')
    assert cpu['status']=='complete'and cpu['result_sha256']==sha(b/'POSTSEAL_RESULTS.json')and r['status']=='pass'and r['count']==32
    from scripts.score_stvg_instance32_v1 import aggregate
    summary,paired=aggregate(d['rows']);assert summary==d['summary']and paired==d['paired'];assert len(d['rows'])==96
    alias={c:f'T{i:03}'for i,c in enumerate(sorted({v['case_id']for v in d['rows']}),1)}
    rows=[dict(v,case_id=alias[v['case_id']])for v in d['rows']];pp=copy.deepcopy(paired)
    for p in pp:p['per_source_delta_pp']={alias[k]:v for k,v in p['per_source_delta_pp'].items()}
    put(out/'RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=pp,GT_read=True,scope='exposed development; fixed geometry, Temporal clock correction only',no_bootstrap=True))
    costs={m:read(b/'execution'/m/'WORKER_EXIT.json')['wall_seconds']for m in ('teacher','fit','readback')}
    put(out/'CONFIG_AND_COST.json',dict(status='complete',count=32,frames=8,side=448,caption_max_new_tokens=160,active_parameters=512,optimizer='L-BFGS',objective='Temporal-Lite shared candidate KL, no L2, full-grid trust',fit_count=28,fallback_count=4,original_baseline_fits_strict_replayed=28,new_candidate_caption_alignment_pairs=t['new_caption_alignment_pairs'],exact_corrected_P0_reused_pairs=t['exact_P0_reused_pairs'],worker_wall_seconds=costs,model_load_seconds=t['model_load_seconds'],teacher_loop_wall_seconds=t['wall_seconds'],time_encoding_proof=t['proof'],GT_use='only after all32 prediction seal; never feedback/interval selection',no_search=True))
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=32,row_count=96,all_metrics_summary_pairs_reaggregated=True,positive_negative_neutral_unknown_preserved=True,source_result_sha256=sha(b/'POSTSEAL_RESULTS.json'),source_prediction_barrier_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),source_root_sha256=sha(b/'ROOT_READBACK.json'),GT_read=True))
    from scripts.export_stvg_lite_batch001 import public_source_text
    bindings=[]
    for name in ('stvg_instance_temporal32_v1.py','score_stvg_instance32_v1.py','report_stvg_instance_temporal32_v1.py','stvg_qwen_uniform_clock002.py','stvg_temporal_lite_fit_v1.py'):
        p=ROOT/'scripts'/name;q=out/'code'/name;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(p.read_bytes()));bindings.append(dict(name=name,source_sha256=sha(p),public_sha256=sha(q)))
    put(out/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings,runtime_sha256=sha(b/'RUNTIME.json')))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));arms=['Frozen','Legacy_Temporal','Corrected_Temporal']
    for j,a in enumerate(arms):ax.bar([i+(j-1)*.23 for i in range(2)],[100*next(v['mean']for v in summary if v['dataset']==ds and v['arm']==a and v['metric']=='m_vIoU')for ds in ('hc2','vidstg')],.23,label=a)
    ax.set_xticks([0,1],['HC development16','VidSTG development16']);ax.set_ylabel('Mean vIoU (%)');ax.set_title('Same candidate inputs and objective; corrected fractional time');ax.legend();fig.tight_layout();fig.savefig(out/'temporal_clock.png',dpi=160);plt.close(fig)
    doc=ROOT/'docs/STVG_INSTANCE_TEMPORAL32_001.md';doc.write_text('''# Matched Temporal fractional-clock correction on32 development sources

The complete Temporal Teacher interface is corrected before the spatial evidence comparison. Original8x448 marked frames, candidate intervals, prompts,160-token caption cap and512-parameter Temporal-Lite objective are unchanged. This is an interface repair, not a claimed innovation.

All28 original eligible Lite fits replay exactly under their original BLAS lineage. Four original ineligible cases retain Native fallback. The corrected run reuses52 caption/alignment pairs only after exact corrected-input and protocol checks, makes60 new pairs and performs28 new Lite fits. An independent CPU replay checks all28 new fits before all32 predictions are sealed and scored. The optimizer is L-BFGS, not Adam.

Relative to the old Temporal baseline, corrected mean vIoU changes by+0.732pp in HC and+0.238pp in VidSTG. Relative to Frozen, changes are+1.847pp and+0.170pp. HC corrected-versus-old signs are4 positive/2 negative/10 unchanged; Vid3/2/11. All96 per-source arm rows, five metrics, nulls and paired deltas are included. No Bootstrap is used.

These32 sources are exposed development:12 are the previously result-stratified diagnostic sources,20 are SHA-ordered additions. This is neither an independent confirmation nor a representative fullset estimate. Geometry remains frozen throughout this group; it does not establish a Spatial improvement. Upcoming E0/E1/E2 all use this same corrected Temporal version and corrected Spatial clock, fixed HC lr .0045 and Vid .027. No learning-rate or prompt sweep is added.

Worker timings, actual reuse/call counts, full anonymous positive/negative/neutral results, exact source bindings and the isolated clock patch are supplied. No videos, private annotations, tensor caches, real Query-caption text or raw Teacher responses are exported. This publication completes only the Temporal32 group; the spatial evidence experiment remains separate.
''')
    paths={str(p.relative_to(ROOT))for p in out.rglob('*')if p.is_file()}|{str(doc.relative_to(ROOT)),'scripts/report_stvg_instance_temporal32_v1.py'}
    put(BASE/'publication/temporal001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]))
    print(json.dumps(dict(status='complete',files=len(paths),costs=costs)))
if __name__=='__main__':main()
