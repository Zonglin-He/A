"""Root reviewed scope descriptions and saved-evidence exports; no model/GT."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy
read,sha=c.read,c.sha
def put(p,d):return c.write(p,d,replace=True)
def main(scope):
    b=c.BASE/scope;out=ROOT/'results/stvg_overnight_consolidation_20261016/2026-10-17'/scope;pub=c.BASE/'publication'/scope
    if scope=='tubedetr_qualification8':
        out.mkdir(parents=True,exist_ok=True);paths=set();q=read(b/'QUALIFICATION_COMPLETION.json');assert q['count']==8 and q['actual_scientific_effect_comparisons']==0
        runtime=read(b/'RUNTIME.json')
        for p,h in runtime['pins'].items():assert sha(ROOT/p)==h,p
        records=[]
        for i,r in enumerate(q['records'],1):
            assert read(b/'probes'/(r['case_id']+'.json'))==r
            for k in ('cached_full_exact','zero_update_exact','actual_nonzero_parameter_step','output_changed','rollback_full_exact','native_coordinate_and_physical_time_interface'):assert r[k]
            records.append(dict(r,case_id=f'R{i:03d}'))
        put(out/'RESULTS.json',dict(q,records=records));put(out/'CONFIG_AND_COST.json',dict(status='pass',count=8,checkpoints=runtime['checkpoints'],native_offsets=1,trainable_parameters=512,step='engineering-only one SGD proposal toward own boxes +0.01, learning rate 1; no Teacher/GT',no_downloads=True,not_TA_cross_domain_weights=True,GT_read=False))
        put(out/'ROOT_INTERFACE_READBACK002.json',dict(status='pass',count=8,all_saved_probes_verified=True,all_runtime_and_weight_hashes_verified=True,confirmation_failed_to_advance=True,scientific_B_effect_not_triggered=True,actual_scientific_effect_comparisons=0,new_model_calls=0,new_GT_reader_calls=0,GT_read=False))
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['case_id'] for r in records],[r['wall_seconds'] for r in records]);ax.set(xlabel='Anonymous probe',ylabel='Interface probe wall seconds',title='TubeDETR native interface: 8 engineering probes, 0 B effect comparisons');fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig)
        text='# TubeDETR native interface qualification (8 sources)\n\nAll eight actual GPU probes passed exact cached/full outputs, zero-update identity, a nonzero engineering parameter/output change and exact rollback on all native frames. Two last-decoder norm3 tensors contain 512 trainable parameters, one native offset. This is an interface test using released domain-specific weights, not TA cross-domain source weights and not a scientific B transfer result.\n\nThe single SGD proposal targets its own predicted boxes plus 0.01, without GT or Teacher evidence. It is not the B optimizer or its acceptance mechanism. Confirmation did not establish B advantage over A, so the protocol condition for a B transfer-effect study was not triggered. Actual scientific effect comparisons: zero. No download, fullset execution or CURRENT promotion. Plot costs cover engineering probes only, not deployment end-to-end cost.\n'
        doc=ROOT/'docs/STVG_NIGHT_TUBEDETR_QUALIFICATION8_002.md';doc.write_text(text);paths.add(str(doc.relative_to(ROOT)))
        names=('stvg_overnight_tubedetr_v1.py','continue_stvg_overnight_tubedetr_v1.py','finalize_stvg_overnight_exports002.py')
    else:
        manifest=read(pub/'EXPORT_MANIFEST.json');paths=set(manifest['paths']);doc=ROOT/'docs'/('STVG_NIGHT_'+scope.upper()+'_001.md');text=doc.read_text();start=text.index('Original acceptance');end=text.index('| Arm |');
        shared='Original acceptance, rejection, rollback, precision, per-domain parameters and final full-frame Student readout remain fixed. These groups use the original four-frame context with global64 + local96 tokens and the original clock, not the separate 32-frame/global512 intervention. Semantic identity/role/full-extent pixel labels remain pending; Student IoU cannot substitute for those labels.\n\n'
        if scope=='confirmation128':
            extra='The candidate was frozen as core B-fullstep before this confirmation. All 128 parents (64/domain) exclude core256 and all sixteen registered historical cohorts; registry absence is not proof of globally unseen data. A and B share the same updater and configurations. B minus Temporal remains positive in both domains (+0.457311 / +0.423790 pp), but B minus matched A is -0.170129 / -0.254425 pp. The formal machine outcome unresolved_spatial_gain therefore means B extra value did not confirm, not that all Spatial benefit disappeared. Method selection stops without a new gate, Prompt, parameter or candidate search; neither A nor B is promoted from this confirmation. TubeDETR scientific B effects and official fullset execution are not triggered.\n\nThe A_Spatial/B_Spatial columns are conditioned readouts using Full observations/references, not new Native-clock spatial-only runs. Source used 128 actual new captures; Temporal used 436 new caption-alignment pairs. WHO: A512, B436 actual calls (76 missing-caption fallback window receipts). Student: A128 fits/118 nonzero/442 accepted updates; B108 fits/94 nonzero/367 accepted updates,20 no-fit sources. These are work counts, not independent additional parents.\n\n'
            put(out/'ROOT_CONFIRMATION_INTERPRETATION002.json',read(b/'ROOT_CONFIRMATION_INTERPRETATION002.json'));put(out/'ROOT_MEASUREMENT_READBACK002.json',read(c.BASE/'ROOT_MEASUREMENT_READBACK002.json'))
        else:
            audit=read(b/'ROOT_COMPONENT_READBACK002.json');assert audit['status']=='pass';put(out/'ROOT_COMPONENT_READBACK002.json',audit)
            if scope=='ablation_noCaption':extra='This fixed deletion removes caption consensus/difference evidence and keeps query relevance. Frame/role quotas and ID/context protection are unchanged. It also restores WHO eligibility for 7 caption-unavailable sources (28 new calls), whereas LocalB retains its pre-existing fallback. Thus this is whole-caption-selector deletion, not a score-only perturbation. On the common-caption secondary subset HC13/Vid12, Candidate minus LocalB is -0.054972 / -2.553279 pp, compared with +0.152285 / -1.894123 pp over all 16/domain. All negative tails remain, including the large Vid decline. No new recipe is selected from this ablation.\n\n'
            else:extra='This fixed deletion removes only the local candidate-priority coverage loop. Root recomputation of all128 actual windows verified unchanged global indices, total token budget, each frame/role quota, ID/context protection, candidate geometry, pixels, Prompt and Student. Missing-caption fallback remains. Candidate minus LocalB is +0.002029 / +0.578743 pp over all16/domain; the common-caption HC13/Vid12 secondary is +0.002498 / +0.771657 pp. This small development deletion is not promoted after failed confirmation; no further tuning or search follows.\n\n'
        doc.write_text(text[:start]+shared+extra+text[end:]);names=('audit_stvg_overnight_components002.py','finalize_stvg_overnight_exports002.py')
    bindings=[]
    for name in names:
        p=ROOT/'scripts'/name;t=out/'code'/name;t.parent.mkdir(exist_ok=True);t.write_text(public_source_text(p.read_bytes()));bindings.append(dict(name=name,source_sha256=sha(p),public_sha256=sha(t)))
    put(out/'ROOT_EXPORT_SOURCE_BINDINGS002.json',dict(status='pass',records=bindings))
    paths|={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths)
    # No raw private roster aliases in any JSON/Markdown export.
    aliases=[r['probe_alias'] for r in read(b/'ROSTER_LOCK.json')['records']]
    for rel in paths:
        p=ROOT/rel
        if p.suffix in ('.json','.md'):
            for alias in aliases:assert '"'+alias+'"' not in p.read_text(),rel
    put(pub/'EXPORT_MANIFEST.json',dict(status='complete',count=8 if scope=='tubedetr_qualification8' else (128 if scope=='confirmation128' else 32),paths=sorted(paths),files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in sorted(paths)]));print(scope,len(paths))
if __name__=='__main__':main(sys.argv[1])
