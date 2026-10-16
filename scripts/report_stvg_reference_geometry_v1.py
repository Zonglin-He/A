"""Export only completed clock audit and geometry decomposition, no raw assets."""
from pathlib import Path
import collections,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_geometry_audit_v1 import BASE,read,put,sha
OUT=ROOT/'results/stvg_spatial_reference_geometry/2026-10-16/audit001'
DOC=ROOT/'docs/STVG_SPATIAL_REFERENCE_GEOMETRY_AUDIT001.md'
SOURCE_NAMES=['stvg_reference_geometry_audit_v1.py','stvg_reference_geometry_clock002.py','stvg_reference_geometry_postseal_v1.py','stvg_reference_clock_control_v1.py','report_stvg_reference_geometry_v1.py','stvg_qwen_uniform_clock002.py','stvg_weightfree_spatial_v1.py']

def export():
    assert not OUT.exists();OUT.mkdir(parents=True)
    clock=read(BASE/'CLOCK_INTERFACE_AUDIT.json');root=read(BASE/'ROOT_GEOMETRY_READBACK.json');diag=read(BASE/'POSTSEAL_GEOMETRY_DIAGNOSIS.json');review=read(BASE/'IDENTITY_VISUAL_REVIEW.json')
    assert clock['status']=='complete'and root['status']=='pass'and diag['count']==review['count']==root['count']==12
    aliases={}
    for cohort in ('development','confirmation'):
        ids=sorted({r['case_id']for r in clock['records']if r['cohort']==cohort})
        for i,c in enumerate(ids,1):aliases[cohort,c]=('D'if cohort=='development'else'C')+f'{i:03}'
    rows=[dict(r,case_id=aliases[r['cohort'],r['case_id']])for r in clock['records']]
    for r in rows:r.pop('job_id',None)
    summary=[]
    for cohort in ('development','confirmation'):
        for branch,clockname,view in [('Temporal',None,None),('Spatial','temporal',0),('Spatial','temporal',1),('Spatial','native',0),('Spatial','native',1)]:
            sub=[r for r in rows if r['cohort']==cohort and r['branch']==branch and(clockname is None or r['clock']==clockname)and(view is None or r['view']==view)]
            summary.append(dict(cohort=cohort,branch=branch,observation_clock=clockname,view=view,count=len(sub),changed_positions=sum(r['affected']for r in sub),old_collapsed=sum(len(set(r['legacy_time_positions']))==1 for r in sub),corrected_collapsed=sum(len(set(r['corrected_time_positions']))==1 for r in sub)))
    put(OUT/'CLOCK_AUDIT.json',dict(status='complete',records=rows,summary=summary,actual_model_tokens_per_second=2,GT_read=False,model_loaded=False,task_performance_effect_measured=False,clock_contracts=read(BASE/'CLOCK_CONTRACTS.json'),scope=clock['scope'],interpretation=clock['interpretation']))
    cleanroot=[]
    for r in root['rows']:cleanroot.append(dict(r,case_id=aliases['development',r['case_id']]))
    put(OUT/'REPLAY_READBACK.json',dict(status='pass',count=12,rows=cleanroot,GT_read=False,AST_contract=read(BASE/'REPLAY_AST_CONTRACT.json'),original_rectangles_historically_cached=False,scope=root['scope']))
    labels={(r['case_id'],r['window_slot']):r for r in review['records']};arows=[]
    for a in diag['anchors']:
        l=labels[a['case_id'],a['window_slot']]
        v=dict(a,case_id=aliases['development',a['case_id']]);v.pop('identity',None)
        v.update({k:l[k]for k in ('observable_region_type','identity_correctness','reason')});arows.append(v)
    counts=[]
    for ds in ('hc2','vidstg'):
        for sign in ('positive','negative','neutral'):
            sub=[r for r in arows if r['dataset']==ds and r['stratum']==sign]
            counts.append(dict(dataset=ds,stratum=sign,anchors=len(sub),observable_region_counts=dict(collections.Counter(r['observable_region_type']for r in sub)),known_GT_anchors=sum(r['known_GT_anchor']for r in sub),exact_historical_intermediates=sum(r['exact_replayed_anchor']for r in sub)))
    put(OUT/'GEOMETRY_DIAGNOSIS.json',dict(status='complete',count=12,anchor_count=len(arows),anchors=arows,summary=counts,GT_read=True,new_fit=False,new_teacher=False,selection='two positive/two negative/two unchanged development sources per domain; prefer references then parent SHA',selection_not_representative=True,visual_review_scope=review['review_scope'],geometry_patch_qualified=False,reasons=review['reasons'],pool_identity_support='unknown_not_exhaustively_reviewed',original_intermediate_geometry_effect='unknown_due_nonexact_replay'))
    costs=read(BASE/'GEOMETRY_PREDICTION_BARRIER.json')
    put(OUT/'COST_AND_RECOVERY.json',dict(status='complete',CPU_replay_wall_seconds=costs['wall_seconds'],replayed_windows=48,replayed_sources=12,new_model_runs_for_this_audit=0,new_fit=0,new_GT_Oracle_fit=0,clock_audit_first_attempt='explained missing Native job lock because exact reused teacher jobs live in original component scope',clock_audit_recovery='strict original job/barrier hash-linked adapter002; original script retained',clock_control='separate 48-request Spatial-only corrected-time control running; not included in audit results'))
    bindings=[]
    for name in ['RUNTIME.json','DESIGN_LOCK.json','ROSTER_LOCK.json','CLOCK_CONTRACTS.json','CLOCK_INTERFACE_AUDIT.json','GEOMETRY_PREDICTION_BARRIER.json','ROOT_GEOMETRY_READBACK.json','POSTSEAL_GEOMETRY_DIAGNOSIS.json','IDENTITY_VISUAL_REVIEW.json']:
        bindings.append(dict(receipt=name,sha256=sha(BASE/name)))
    put(OUT/'SOURCE_BINDINGS.json',dict(status='pass',receipts=bindings,scope='private input/source chain digests; no private payload paths or media exported'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    selected=[r for r in summary if r['view']!=1]
    names=[f"{r['cohort'][:3]} {r['branch']}"+(f" {r['observation_clock']}"if r['branch']=='Spatial'else'')for r in selected]
    axes[0].barh(names,[100*r['changed_positions']/r['count']for r in selected]);axes[0].set_xlim(0,100);axes[0].set_xlabel('Requests/views with changed temporal-axis positions (%)');axes[0].set_title('Interface audit, actual 2 Hz model')
    cats=['background_or_mixed_region','different_entity_visible','plausible_target_part'];tot=collections.Counter(r['observable_region_type']for r in arows)
    axes[1].bar(['Background / mixed','Different entity','Plausible target part'],[tot[c]for c in cats],color=['#b36b2c','#b94444','#4a7ca5']);axes[1].set_title('22 adopted references: visual region type');axes[1].set_ylabel('Observed anchors, stratified 12-source panel');axes[1].tick_params(axis='x',labelrotation=18)
    fig.tight_layout();fig.savefig(OUT/'audit.png',dpi=160);plt.close(fig)
    text='''# Spatial-reference decomposition: completed audit, conditional geometry patch stopped

The twelve-source panel does not qualify the proposed historical R0/R1/R2 geometry comparison. All 48 windows fail full-pool exact replay; none of the 22 actually adopted anchor component masks/boxes exactly matches the controlled replay. Original rectangles and complete GrabCut foregrounds were not cached. Replayed geometry is retained privately, but it cannot be substituted as if it were the same historical candidate.

The panel contains two positive, two negative and two unchanged development sources per domain. It was selected from already exposed development128 using sealed Full-minus-Temporal signs, preferring actual references and then parent SHA. It excludes confirmation64, but is deliberately result-stratified, not an unbiased effectiveness estimate.

All twelve actual scene panels were viewed with original masks and their own Query. Of the 22 adopted regions, 10 are background/mixed, 6 visibly select another entity, and 6 are plausible target parts. These are single-reviewer scene observations, not exhaustive semantic identity annotations. No complete-object correctness rate is claimed. Negative-source anchors include three plausible target parts, two visibly different entities and five background/mixed regions. Positive cases also contain poor regions: positive final vIoU does not establish correct teacher identity.

The evidence therefore does not establish largest-component truncation as the dominant cause. We stop the geometry patch rather than fitting unlinked reconstructed rectangles or using GT to select candidates. Whether full foreground would help remains unknown; this result does not show it cannot help. The pool's best identity support remains unverified. Exact postseal GT anchor IoUs and unknowns are retained in the anonymous rows, without annotation coordinates.

## Time interface audit

The final Lite paths used the legacy Qwen seconds dtype; only the completed visual-budget P0 had the correction. CPU qualification invokes the installed get_rope_index and the isolated correction at the actual model configuration of 2 tokens/second. The 21 tests cover fractional, near-integer and integer seconds with 2/3/4 temporal grids. Saved original FPS and frame counts determine each actual request's temporal-axis positions; this is not a new model inference or a task-performance measurement.

| Cohort / input | Audited views or requests | Changed positions |
|---|---:|---:|
| Development Temporal | 444 | 374 |
| Confirmation Temporal | 224 | 180 |
| Development Spatial event context, QA-clock | 512 | 292 |
| Development Spatial event context, Native-clock | 512 | 248 |
| Confirmation Spatial event context, QA-clock | 256 | 100 |
| Confirmation Spatial event context, Native-clock | 256 | 124 |

All 1,536 local five-frame Spatial views have identical old/corrected temporal-axis positions. Their short duration falls inside the model's legitimate 2 Hz quantization; repeated positions here are not evidence of the integer-dtype defect. This is why a fractional seconds value alone cannot classify a request as affected. The original grid/FPS receipts remain private; full anonymous position diagnostics are exported.

The development Native evidence is an exact reuse from the completed component study, not a new job lock in the Lite Native reference directory. The first audit attempt stopped on that provenance lookup; adapter002 verifies the original job-lock and teacher-barrier hashes before reading them. No teacher or Student GPU work was repeated for this audit.

## Fixed next control and boundaries

A separate finite 48-request Spatial WHO interface control uses the identical historical candidate videos, prompts, IDs, weights, generation cap and physical centers for these twelve sources, with only fractional-time handling corrected. Historical Temporal-Lite output intervals stay fixed. This is a Spatial interface control, not a fully corrected Temporal method or a frame-selection improvement. Changed requests receive fresh responses; no legacy response is relabeled as corrected. Its results will be published separately after seal/readback.

HC lr .0045, Vid .027, Temporal-Lite and the episodic 1792-parameter Spatial interface remain fixed. No new model, candidate mechanism, geometry Oracle fitting, learning-rate/prompt search, Bootstrap, CURRENT promotion, or official full-set run was performed. An input fix or removal of a connected-component filter is not by itself a paper-level algorithm contribution.

Implementation/protocol and all positive/negative/neutral/unknown anonymous diagnostics are included. Private frames, GT coordinates, weights, tensors, actual Query/caption text, raw teacher responses and personal conversation are excluded.
'''
    DOC.write_text(text)
    put(OUT/'ROOT_PUBLIC_READBACK.json',dict(status='pass',clock_records=len(rows),geometry_sources=12,geometry_anchors=len(arows),all_clock_scalar_rows_exact=True,all_anonymous_anchor_scalar_rows_exact=True,nulls_preserved=True,visual_region_counts=dict(tot),GT_Oracle_fits=0,geometry_patch_executed=False,whole_research_FINAL=False))
    paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()}|{str(DOC.relative_to(ROOT)),'protocols/stvg_spatial_reference_geometry_v1.md'}|{'scripts/'+n for n in SOURCE_NAMES}
    put(BASE/'publication/audit001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]))
    print(json.dumps({'status':'complete','files':len(paths),'anchors':len(arows),'region_counts':dict(tot)}))

if __name__=='__main__':export()
