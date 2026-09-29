"""Paired S0/S0.5 diagnostic and source-level uncertainty, without promotion."""
import sys,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_spatial_propagation_s05_v1 import OUT,S0,verify
from scripts.summarize_tastvg_spatial_s0_components_v1 import calculate
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats
from scripts.audit_tastvg_spatial_expansion_public_v1 import run as audit

def run():
    lock=verify();rows=read(OUT/'ROWS.json');old=read(S0/'ROWS.json');s=read(OUT/'SUMMARY.json');comp=calculate(rows);oldcomp=calculate(old)
    write(OUT/'COMPONENTS.json',comp);comparison={}
    for group in ['corruption','clean']:
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']=='clean')];oo={(r['parent'],r['condition']):r for r in old};by=collections.defaultdict(list)
        for r in rr:by[r['parent']].append(r)
        src=[]
        for p,seq in sorted(by.items()):
            paired=[r for r in seq if r['observed_gain_at_s_oracle'] is not None and r['unobserved_gain_at_s_oracle'] is not None]
            rec=dict(parent=p,whole_oracle_gain=np.mean([r['expanded_gain'] for r in seq]),union_gain=np.mean([r['union_oracle_s']-r['native_s'] for r in seq]),oracle_minus_s0=np.mean([r['expanded_gain']-oo[p,r['condition']]['expanded_gain'] for r in seq]),union_minus_s0=np.mean([r['union_oracle_s']-oo[p,r['condition']]['union_oracle_s'] for r in seq]),unobserved_paired=None,unobserved_minus_s0=None)
            if paired:
                rec.update(unobserved_paired=np.mean([r['unobserved_gain_at_s_oracle'] for r in paired]),unobserved_minus_s0=np.mean([r['unobserved_gain_at_s_oracle']-oo[p,r['condition']]['unobserved_gain_at_s_oracle'] for r in paired]))
            src.append(rec)
        assert sum(x['unobserved_paired'] is not None for x in src)==12
        comparison[group]=dict(source_rows=src,metrics={key:stats([r[key] for r in src]) for key in src[0] if key!='parent'},fixed_unobserved_steps={str(j):stats([np.mean([r['unobserved_step_gains'][j] for r in seq if r['unobserved_step_gains'][j] is not None]) for seq in by.values()]) for j in range(4)},unobserved_only_oracle=stats([np.mean([max(v for v in r['unobserved_step_gains'] if v is not None) for r in seq]) for seq in by.values()]),direct_dense_target_quality={})
        dense=[seq for seq in by.values() if any(r['dense_unobserved_s'] is not None for r in seq)]
        for key in ['dense_unobserved_s','native_dense_unobserved_s']:
            comparison[group]['direct_dense_target_quality'][key]=stats([np.mean([r[key] for r in seq if r[key] is not None]) for seq in dense])
    write(OUT/'COMPARISON.json',comparison);write(OUT/'PUBLIC_AUDIT.json',audit(OUT))
    alloc=[read(f) for f in (OUT/'allocations').glob('*.json')];resource=dict(gpu_process_seconds=sum(a['seconds'] for a in alloc),allocations=len(alloc),failures=sum(a['status']=='failed' for a in alloc),new_expert_calls=0,new_backbone_captures=0,backward_calls=288,full_reinsertion_cells=2)
    write(OUT/'RESOURCES.json',resource)
    counts=collections.Counter(r['propagation']['reason'] for r in rows)
    diagnostics=dict(propagation_status=dict(counts),propagated_frames=sum(r['propagation']['propagated_frames'] for r in rows),sparse_valid_frames=sum(r['valid_evidence_frames'] for r in rows),dense_valid_frames=sum(r['dense_valid_frames'] for r in rows),loss_before=np.mean([r['diagnostics'][0]['loss_before'] for r in rows]),loss_after=np.mean([r['diagnostics'][-1]['loss_after'] for r in rows]),loss_decreased_cells=sum(r['diagnostics'][-1]['loss_after']<r['diagnostics'][0]['loss_before'] for r in rows))
    write(OUT/'DIAGNOSTICS.json',diagnostics)
    source=read(S0/'PROVENANCE.json');write(OUT/'PROVENANCE.json',dict(backbone='TA-STVG',source_checkpoint=source['source_checkpoint'],source_model_state_sha256=source['source_model_state_sha256'],sources=16,queries=16,cells=96,conditions=lock['conditions'],cohort=source['cohort'],GT_scope='same 16 historically exposed keys, post-seal scoring only',reused_expert_revision='3fee777d49ee9276eac51ea3e5f9b69e81d09be6',pins=lock['pins'],prior_barrier_hashes=lock['prior_barriers'],production_changed=False))
    pp=lambda x:f'{100*x:.4f}'
    ci=lambda x:'['+', '.join(pp(v) for v in x['ci95'])+']'
    lines=['# S0.5 — Sparse-to-dense spatial evidence propagation','', 'Completed / audited candidate-support diagnostic on exposed development data. No selector, persistent adapter or OPD was run.','', '| Diagnostic | S0 corruption | S0.5 corruption | S0 clean | S0.5 clean |','|---|---:|---:|---:|---:|']
    oldsum=read(S0/'SUMMARY.json')
    for label,fn in [('Unobserved gain, matched12, pp',lambda sm,c: c['unobserved_gain']['mean']),('Observed gain, matched12, pp',lambda sm,c:c['observed_gain']['mean']),('Whole-tube oracle gain,16 sources, pp',lambda sm,c:sm['metrics']['expanded_gain']['mean']),('Six-layer union gain,16 sources, pp',lambda sm,c:sm['metrics']['union_oracle_s']['mean']-sm['metrics']['native_s']['mean']),('Unobserved gain, all16, pp',lambda sm,c:sm['metrics']['unobserved_gain_at_s_oracle']['mean']),('Fixed step3 whole-tube gain, pp',lambda sm,c:sm['metrics']['step3_gain']['mean'])]:
        vals=[fn(oldsum['corruption'],oldcomp['corruption']),fn(s['corruption'],comp['corruption']),fn(oldsum['clean'],oldcomp['clean']),fn(s['clean'],comp['clean'])];lines.append('| '+label+' | '+' | '.join(pp(v) for v in vals)+' |')
    lines+=['','## Paired inference and negative cases','']
    for group in ['corruption','clean']:
        m=comparison[group]['metrics'];lines+=[f"{group}: matched12 unobserved gain {pp(m['unobserved_paired']['mean'])}pp CI{ci(m['unobserved_paired'])}; change vs S0 {pp(m['unobserved_minus_s0']['mean'])}pp CI{ci(m['unobserved_minus_s0'])}. Whole-tube change vs S0 {pp(m['oracle_minus_s0']['mean'])}pp CI{ci(m['oracle_minus_s0'])}; union change {pp(m['union_minus_s0']['mean'])}pp CI{ci(m['union_minus_s0'])}.",'',f"Fixed unobserved gains at steps1/2/3, all16: {' / '.join(pp(comparison[group]['fixed_unobserved_steps'][str(j)]['mean']) for j in [1,2,3])}pp. Unobserved-only oracle upper bound: {pp(comparison[group]['unobserved_only_oracle']['mean'])}pp (different selection rule; not the primary diagnostic).",'']
    lines+=['| Source | S0.5 whole-tube oracle gain pp | Change vs S0 pp | Union gain pp | Matched unobserved gain pp |','|---|---:|---:|---:|---:|']
    for r in comparison['corruption']['source_rows']:lines.append(f"| Q{r['parent']+1:02} | {pp(r['whole_oracle_gain'])} | {pp(r['oracle_minus_s0'])} | {pp(r['union_gain'])} | {pp(r['unobserved_paired']) if r['unobserved_paired'] is not None else 'NA'} |")
    lines+=['','## Actual configuration and interpretation limits','', 'Original16 C3 VidSTG sources, one query each, same Vid-source TA checkpoint and clean/five source-hash seed0 transient5% corruptions.96 cells. Same frozen H and five Sa2VA masks, zero new expert calls. Fixed .5 global foreground/background cosine contrast + .5 maximum foreground-reference affinity on H_app. Area-projected reference masks (>=.5), fixed reference-class-mean midpoint E threshold, grid-mask enclosing boxes on unobserved frames; original sparse boxes retained at references. This E-to-box readout was specified before scoring to complete the attachment. It is a heuristic; local self matches affect threshold calibration. No trained memory/HTR reproduction claim.','', 'Reuse the unchanged S0 equal-valid-frame L1+GIoU loss, three .004-original-visual-norm H_app-only steps and dynamic native suffix. Adding dense frames also changes the fraction of loss contributed by references. Frozen text/motion/weights. Native B0 retained, so oracle non-harm is structural and does not establish online selector safety. The experiment tests this complete propagation/readout implementation, not only the mathematical affinity formula.','', 'After all96 predictions/evidence sealed, retain only the same16 old GT entries for scoring. sIoU uses all GT-valid sampled frames; choose one entire tube by sIoU and then read its observed/unobserved metrics. Matched12 reproduces S0 denominator; all16 is separate. Five corruption conditions averaged inside each source, then source macro;10000 source-bootstrap draws seed20260929. Descriptive CIs on repeatedly exposed development sources, no untouched-generalization claim.','',f"Propagation coverage: {dict(counts)}; newly supervised unobserved frame-cells {diagnostics['propagated_frames']}; original valid sparse frame-cells {diagnostics['sparse_valid_frames']}; dense total {diagnostics['dense_valid_frames']}. Loss {diagnostics['loss_before']:.6f} → {diagnostics['loss_after']:.6f}, decreases in {diagnostics['loss_decreased_cells']}/96 cells. This loss magnitude uses a different frame denominator from S0.",'']
    for group in ['corruption','clean']:
        q=comparison[group]['direct_dense_target_quality'];lines.append(f"{group}: direct propagated target-box sIoU {pp(q['dense_unobserved_s']['mean'])}% vs native {pp(q['native_dense_unobserved_s']['mean'])}% on the identical GT-valid, dense-target-valid unobserved frames, source count {q['dense_unobserved_s']['n']}. Diagnostic target quality, not a new inference arm; excludes unavailable target frames.")
    lines+=['','## Verification and resources','',f"GPU process {resource['gpu_process_seconds']:.2f}s including initialization,288 suffix backward calls,2 genuinely edited full-native reinsertion checks,0 new expert calls,0 new H captures. Six CPU tests. AUDIT.json:960 dual metrics and576 independent loss checks. PROPAGATION_AUDIT.json independently reconstructs area projection, affinities, thresholded masks and propagated boxes in NumPy. PUBLIC_AUDIT.json reconstructs saved scalar summaries. Prior model-constructor warnings are unchanged; loaded source state hash and exact B0 are checked. No failed cells silently dropped.",'', 'S1 remains a future proposal. Leave-one-reference-out generation creates fold-specific tubes: scores cannot be averaged as if identical step indices were identical candidates without defining that contract. No S1 implementation, additional hyperparameter search, temporal experiment or production promotion.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write(OUT/'COMPLETION.json',dict(status='completed',measurement='audited',time=time.time(),files={n:sha(OUT/n) for n in ['ROWS.json','SUMMARY.json','COMPARISON.json','AUDIT.json','PROPAGATION_AUDIT.json','REPORT.md']}))
    print({g:{k:(v['mean'],v['ci95']) for k,v in comparison[g]['metrics'].items()} for g in ['corruption','clean']})

if __name__=='__main__':run()
