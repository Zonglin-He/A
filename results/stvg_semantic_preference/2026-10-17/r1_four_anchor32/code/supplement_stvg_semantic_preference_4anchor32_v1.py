"""Postseal saved-scalar decision and publication evidence only; no new GT/model reader."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_semantic_preference_4anchor32_v1';OUT=ROOT/'results/stvg_semantic_preference/2026-10-17/r1_four_anchor32'
def main():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    d=read(OUT/'RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'ROOT_SCALAR_READBACK.json')['checks']==2250
    decision=dict(status='completed',measurement='valid_with_independent_saved_scalar_and_state_readback',count=32,cohort='previously_exposed_development_HC16_Vid16',new_mechanism_increment='not_established_in_scope',four_anchor_correction='does_not_close_Direct_end_to_end_vIoU_gap',promote_CURRENT_or_PAPER=False,new_model_or_GT_reader=False,comparisons=[])
    for p in d['paired']:
        if p['metric']=='m_vIoU' and p['comparison']=='PREF_minus_GEOM':
            lo,hi=p['paired_source_t95_interval_pp'];assert lo<0<hi
            decision['comparisons'].append(dict(dataset=p['dataset'],comparison=p['comparison'],mean_delta_pp=p['mean_delta_pp'],paired_source_t95_interval_pp=[lo,hi],positive=p['positive'],negative=p['negative'],neutral=p['neutral'],unknown=p['unknown']))
    cross=read(OUT/'DIRECT_AND_ONE_ANCHOR_COMPARISONS.json');assert cross['baseline_exact_scalar_checks']==192
    decision['comparisons'] += [p for p in cross['records'] if p['metric']=='m_vIoU' and p['comparison'] in ['PREF4_minus_Direct4','PREF4_minus_PREF1']]
    decision['remaining_Direct_confounders']=['fixed candidate atoms versus continuous Direct coordinates','fixed-support distillation versus on-policy32 sampled-action OPD','preference-numbered-image Prompt and two likelihood orders versus one coordinate generation']
    write(OUT/'DECISION.json',decision)
    doc=ROOT/'docs/STVG_SEMANTIC_PREFERENCE_R1_FOUR_ANCHOR32_001.md';text=doc.read_text();assert '## Measured decision' not in text
    text+='\n## Measured decision\n\nFour-anchor PREF vIoU HC27.357912%/Vid11.974958% remains below revised Direct HC28.580257%/Vid14.572580% (−1.222345/−2.597622pp). Versus the previous single anchor it changes +.115417/−.532209pp. The same-support GEOM control is27.363192%/12.274769%; PREF−GEOM −.005279/−.299810pp, with both paired source t95 intervals spanning0. Therefore this experiment does not establish a semantic-weight benefit, and adding anchors did not close the end-to-end gap. These32 are exposed development and are not new confirmation.\n\nDense common-time sIoU PREF is50.107393%/32.782728%; revised Direct52.108541%/31.878854%. Direct is not better on every metric: Vid dense geometry favors PREF, despite its lower final vIoU. The result cannot be reduced to a universal semantic-versus-geometry verdict because the remaining interfaces/action distributions differ. All signs/tails and two no-fit sources remain; no tuning, method promotion or new search is triggered. Same-reference parameter-free D2 is17.395617%/8.542736% vIoU in this sparse candidate-reference setting.\n'
    doc.write_text(text)
    cost=read(OUT/'COST.json');cost['actual_new_Source_capture_calls']={a:read(BASE/'arms'/a/'trials/G32/STATUS.json')['source_capture_calls'] for a in ['GEOM','PREF']};cost['GEOM_profile_receipt_unavailable_due_to_preserved_filename_conflict']=True;cost['historical_one_anchor_profile_not_relabelled_as_this_run']=True;write(OUT/'COST.json',cost,mutable=True)
    p=OUT/'code'/Path(__file__).name;p.write_text(public_source_text(Path(__file__).read_bytes()))
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_semantic_preference_student32_v1.md','protocols/stvg_semantic_preference_student32_audit002.md','protocols/stvg_semantic_preference_4anchor32_v1.md','protocols/stvg_semantic_preference_4anchor32_recovery002.md'};verify_privacy(paths)
    write(BASE/'publication/r1_four_anchor32/EXPORT_MANIFEST.json',dict(status='complete',count=32,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]),mutable=True)
    write(BASE/'ROOT_DECISION_READBACK.json',dict(status='pass',count=32,source_paired_saved_scalars_only=True,baseline_exact_scalar_checks=192,GT_read=False))
if __name__=='__main__':main()
