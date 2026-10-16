"""Complete anonymous original-candidate-pool diagnosis, no new fit."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
OUT=ROOT/'results/stvg_instance_evidence32/2026-10-16/pool001'
DOC=ROOT/'docs/STVG_INSTANCE_POOL12_001.md'
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);d=read(BASE/'POOL12_POSTSEAL_DIAGNOSIS.json');v=read(BASE/'POOL12_VISUAL_REVIEW.json');bar=read(BASE/'POOL_INPUT_BARRIER.json')
    assert d['barrier_sha256']==sha(BASE/'POOL_INPUT_BARRIER.json')and d['window_count']==v['windows_reviewed']==bar['windows']==48
    aliases={c:f'P{i:03}'for i,c in enumerate(sorted({r['case_id']for r in d['records']}),1)};rows=[]
    for w in d['records']:
        non=[x['GT_iou']for x in w['candidates']if x['role']=='proposal'and x['GT_iou']is not None];assert (max(non)if non else None)==w['best_nonSource_iou'];assert w['candidates'][0]['GT_iou']==w['Source_iou']
        rows.append(dict(w,case_id=aliases[w['case_id']]))
    summary=[]
    for ds in ('hc2','vidstg'):
        sub=[r for r in rows if r['dataset']==ds];known=[r for r in sub if r['known_GT']]
        summary.append(dict(dataset=ds,windows=len(sub),known_GT_windows=len(known),unknown_GT_windows=len(sub)-len(known),nonSource_ge_0_5=sum(r['best_nonSource_iou']is not None and r['best_nonSource_iou']>=.5 for r in known),Source_ge_0_5=sum(r['Source_iou']>=.5 for r in known),nonSource_better_than_Source=sum(r['best_nonSource_iou']is not None and r['best_nonSource_iou']>r['Source_iou']for r in known),mean_best_nonSource_iou=math.fsum(r['best_nonSource_iou']for r in known if r['best_nonSource_iou']is not None)/len(known)if known else None))
    put(OUT/'POOL_DIAGNOSIS.json',dict(status='complete',count=12,window_count=48,candidate_count=sum(len(r['candidates'])for r in rows),records=rows,summary=summary,GT_read=True,GT_use='offline exact physical observation positions only; never reference construction/ID choice/training',identity_annotation='unknown for candidate identity; scalar IoU threshold is not identity proof',selection='already exposed sign-stratified diagnostic12; not a representative estimate',no_Oracle_fit=True))
    put(OUT/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=12,window_count=48,all_candidate_scalar_rows_exact=True,nulls_preserved=True,actual_private_pool_views_reviewed=12,no_proposal_regeneration=True,source_barrier_sha256=sha(BASE/'POOL_INPUT_BARRIER.json'),postseal_diagnosis_sha256=sha(BASE/'POOL12_POSTSEAL_DIAGNOSIS.json'),visual_review_sha256=sha(BASE/'POOL12_VISUAL_REVIEW.json'),GT_read=True))
    decision=read(BASE/'EVIDENCE_MODIFICATION_LOCK.json');put(OUT/'NEXT_EVIDENCE_DESIGN.json',decision)
    from scripts.export_stvg_lite_batch001 import public_source_text
    source=[]
    for name in ('stvg_instance_evidence32_v1.py','stvg_instance_temporal32_v1.py','report_stvg_instance_pool12_v1.py','publish_stvg_instance_evidence32_v1.py'):
        p=ROOT/'scripts'/name;q=OUT/'code'/name;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text(p.read_bytes()));source.append(dict(name=name,source_sha256=sha(p),public_sha256=sha(q)))
    put(OUT/'SOURCE_BINDINGS.json',dict(status='pass',records=source,runtime_sha256=sha(BASE/'RUNTIME.json'),roster32_sha256=sha(BASE/'ROSTER_LOCK.json'),design_sha256=sha(BASE/'DESIGN_LOCK.json')))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));x=range(2);ax.bar([i-.18 for i in x],[r['Source_ge_0_5']for r in summary],.36,label='Source');ax.bar([i+.18 for i in x],[r['nonSource_ge_0_5']for r in summary],.36,label='Any nonSource');ax.set_xticks(list(x),['HC','VidSTG']);ax.set_ylabel('Exact observation windows with box IoU >= .5');ax.set_title('Historical pool geometry; unknown positions excluded');ax.legend();fig.tight_layout();fig.savefig(OUT/'pool_coverage.png',dpi=160);plt.close(fig)
    DOC.write_text('''# Whole-pool instance-evidence diagnosis

All actual candidate IDs in the twelve previously reviewed development sources are now checked, using cached historical boxes, masks, RGB and teacher ID bindings. No historical Selective Search/GrabCut result is regenerated. The twelve scene/pool panels were actually viewed, including positive, negative and unchanged sources.

Among48 windows,25 have exact physical-position GT and23 remain unknown. No nonSource candidate reaches box IoU .5 on those25 windows; Source reaches it on7. These are geometric coverage statistics, not semantic identity labels: visual review finds a few plausible correct instances at lower IoU, and annotation extent can differ from visible-instance extent. Do not infer that every nonSource candidate is another object or that all48 windows lack the referent.

Frequent nonSource candidates depict background, clothing/body fragments, other entities or mixed regions. Source often supplies the most recognizable instance hypothesis. There are also sources where Source itself follows the wrong person/region; reorganizing existing crops cannot create a missing whole-object hypothesis. All candidate-level scalar IoUs, selected IDs, presence, nulls and window strata are included anonymously. The panel is result-stratified and previously exposed, not a representative confirmation set.

## One prospective change, not a prompt search

The next fixed32-source experiment tests Source-seeded RGB foreground as one additional instance hypothesis within the same nine-candidate budget. The old last proposal slot is replaced, other IDs stay fixed. This is a new prospective candidate, not a reconstructed historical foreground. It uses no new weights, labels, detector or tracker. Original three-iteration GrabCut and local DIS are reused; the complete foreground is retained. Expansion1.25 and central-half sure-foreground seed are locked once, with no sweep.

This hypothesis inherits the Source position prior, can merge background, can delete a useful old last-slot proposal and cannot reliably discover a referent outside a wrong Source region. It is not claimed as independent object discovery or as a paper contribution before evidence. Invalid masks retain the original pool and remain in the denominator. Original masks/rectangles/seeds and new intermediate outputs are preserved privately.

Before E0/E1/E2, the complete Temporal Teacher baseline is being corrected under identical original8x448 sampling, prompts, candidate intervals,512-parameter objective and CPU lineage. E0 and E1 then use the same corrected Temporal/Spatial time version and physical centers; E2 shares exactly E1 references and changes only to original validated OPD. HC lr .0045/Vid .027 stay fixed. All no-reference/no-update/failure cases remain. No further Full-scene/Pairwise gate, Teacher replacement, LR search, Bootstrap, CURRENT change or full official evaluation is launched.

This publication completes the pool-diagnosis group only. The prospective proposal and E0/E1/E2 task effectiveness are not results yet. Private media/annotation coordinates/weights/tensors/Query-caption/raw teacher outputs are excluded.
''')
    paths={str(p.relative_to(ROOT))for p in OUT.rglob('*')if p.is_file()}|{str(DOC.relative_to(ROOT)),'protocols/stvg_instance_evidence32_v1.md','scripts/report_stvg_instance_pool12_v1.py','scripts/publish_stvg_instance_evidence32_v1.py'}
    put(BASE/'publication/pool001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]))
    print(json.dumps({'status':'complete','summary':summary,'files':len(paths)}))
if __name__=='__main__':main()
