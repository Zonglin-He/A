"""Cached-only diagnosis report; measured net correction distinct from code path."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.diagnose_stvg_dino_reference_path_cache32_v1 import D,P,PRE,HF,verify


def main():
    import matplotlib.pyplot as plt
    p=verify();r=read(P/'DIAGNOSIS.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    code=(ROOT/HF).read_text();checks={
        'reference_dependent_positional_embedding':'get_sine_pos_embed(reference_points_input' in code,
        'box_size_scaled_deformable_sampling':'sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5'in code,
        'iterative_logit_reference_addition':'tmp + torch.special.logit(reference_points, eps=1e-5)'in code,
        'forward_refinement_despite_detach':'reference_points = new_reference_points.detach()'in code,
        'score_from_hidden_text_dot_product':'output = vision_hidden_state @ text_hidden_state.transpose(-1, -2)'in code}
    assert all(checks.values())
    put(P/'CODE_PATH_AND_LIMITS.json',dict(status='source_checked_not_layer_measured',installed_HF_SHA256=sha(ROOT/HF),decoder_layers=6,checks=checks,
        reference_effect='Reference changes positional embeddings and spatial sampling center/extent; subsequent box predictions feed next layer.',
        classification='Final target score comes from final hidden state dot text embeddings, not from output-box overlap.',
        detach='Stops gradient through iterative reference; does not stop forward coordinate refinement. No gradient updates here.',
        wrapper='Uses final normalized intermediate hidden, penultimate refined reference and final bbox/class heads; ordinary native replay and clones were qualified in prior actual run.',
        unknown=['native initial900 reference','per-layer reference and sampling locations','per-layer hidden/text score','first-layer versus cumulative loss','overshoot versus recovery path','frozen/native-reference counterfactual'],
        native_final_proxy_not_initial=True,native_probe_original_image_clipping_and_roundtrip=True,no_new_forward=True))
    put(P/'CONFIGURATION.json',dict(count=32,per_domain=16,prior_commit='a3de927f8655da16a83407ba36b8b0f4c8df2578',threshold=.35,
        unchanged_source_and_E_MBR=True,only_cached_scalars_and_predictions=True,GT_read=False,model_calls=0,new_predictions=0,
        exact_native_slot_score_match_required=True,fallback_placeholder_ID_excluded=True,CPU_seconds_cap=600,
        current_registry_SHA256=sha(ROOT/'methods/CURRENT_METHOD.json'),source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/dino_conditional32/CONFIGURATION.json')['source_checkpoints']))
    lines=['# Why native high-score content stays low at student references','','Original32 exposed development, cached-only. No model/decoder/processor/weights/media/GT access, no threshold or roster changes, no new predictions. The previous matched intervention held references and masks fixed while changing content. This diagnosis varies stored references under the same high content and attention-mask slot, and measures decoder NET input-to-final changes. It does not invent per-layer traces.','','## Available evidence','','| Domain | Native final box recovered / active anchor frames | Matched references | Admitted | Input/native final IoU median | Output/native final IoU median |','|---|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        a=g['all_matched'];lines.append(f'| {ds} | {g["native_final_frames_recovered"]}/{g["native_frames"]} | {a["references"]} | {a["admitted"]} | {a["input_native_final_IoU"]["quantiles"]["median"]:.6f} | {a["output_native_final_IoU"]["quantiles"]["median"]:.6f} |')
    lines+=['','Native box recovered only by matching original native slot plus target score exactly. Fallback placeholder IDs and missing native Top3 slots remain unknown. One original-cohort full900 cache verifies4 matching frames; no new cohort added. Native FINAL candidate box is neither GT nor INITIAL native proposal; the original probe clipped xyxy to image bounds and converted back to cxcywh, so it is not the raw native bbox tensor. The one full900 case reproduces this original postprocessing exactly. Relative offsets below describe distance to this final detection, not exact displacement from the native starting reference.','','## Reference offsets and net correction','','| Domain | Input overlap bin | References | Admitted | Conditional score median | Output overlap median | Net overlap gain mean pp |','|---|---|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        for k,a in g['input_IoU_bins'].items():lines.append(f'| {ds} | {k} | {a["references"]} | {a["admitted"]} | {a["conditional_score"]["quantiles"]["median"]:.6f} | {a["output_native_final_IoU"]["quantiles"]["median"]:.6f} | {a["net_IoU_gain_pp"]["mean"]:+.6f} |')
    lines+=['','Center offset is normalized by native final width/height; log-size offset is Euclidean norm of log width/height ratios. Separate descriptive bins are in DIAGNOSIS.json. They can covary and are not independent causal percentages. Same-query candidates are not independent statistical samples.','','## Fixed content AND mask reference variation','', '| Domain | Groups with >=2 distinct refs | Groups crossing .35 | Query mean nearer-minus-farther score [95% CI] |','|---|---:|---:|---:|']
    for ds,g in r['groups'].items():
        f=g['fixed_content_and_mask'];v=f['query_near_minus_far_score'];ss='unknown'if v['mean']is None else f'{v["mean"]:+.6f} [{v["CI95"][0]:+.6f},{v["CI95"][1]:+.6f}]';lines.append(f'| {ds} | {f["groups"]} | {f["crosses035"]} | {ss} |')
    lines+=['','Within each query/frame/high-content/mask group, deduplicate FP32 references, compare largest versus smallest native-final input IoU. Aggregate groups within query first; intervals bootstrap queries. This directly observes reference sensitivity with content and mask held fixed; all decoder responses are still full six-layer responses. It cannot allocate that sensitivity to a particular layer.','','## Geometric recovery versus score recovery','','| Domain | Final overlap >=.75: admitted / refs | Final overlap >=.9: admitted / refs |','|---|---:|---:|']
    for ds,g in r['groups'].items():
        a,b=g['recovered_ge075'],g['recovered_ge09'];lines.append(f'| {ds} | {a["admitted"]}/{a["references"]} | {b["admitted"]}/{b["references"]} |')
    lines+=['','These are agreements with a native detector box, not correct-identity or GT claims. Native geometry and conditional classification are different outputs: recovering coordinates need not restore the original hidden/text compatibility. Conversely, final proximity alone cannot establish an early-versus-late decoder failure.','','## What source inspection establishes, and what it cannot','','The installed decoder recomputes sine positional embeddings from current references; deformable visual attention samples around the reference center with its width/height scaling. Each layer applies a bbox increment in logit coordinates and forwards the new reference into the next layer. Final target score is hidden-state/text dot product followed by sigmoid/target-token averaging. A high-score content ROW is not the native final hidden state. Replacing the reference therefore changes the actual visual and positional processing, rather than attaching a fixed confidence to a content vector. These are source-code facts, not measured attention maps.','','`.detach()` affects backward derivatives, not forward coordinate refinement. It cannot by itself explain missing feedback in this no-gradient detector evaluation. Prior native replay/clone qualification checks the wrapper on native states; it does not qualify arbitrary student references.','','The saved pilot contains only final conditional boxes/scores, selected slots, actions, control hashes and native top scores. Native initial references, DINO six-layer boxes/logits/hidden states/sampling points are absent. Other cached six-layer Student boxes are not DINO traces. We cannot tell whether score was low at layer1, decayed cumulatively, or briefly recovered then fell. No new forward was run to fill this gap.','','Decision: retain the previous stop before OPD. This diagnosis narrows geometry and interface mechanisms but does not authorize lowering .35, tuning a layer or starting another experiment. A layerwise counterfactual remains unmeasured. Missing native frames/GT stay unknown. Current registry and all prior predictions preserved.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('See REPORT.md, DIAGNOSIS.json, WITHIN_FIXED_CONTENT_MASK.json, CACHE_AVAILABILITY.json, CODE_PATH_AND_LIMITS.json and ROOT_INDEPENDENT_AUDIT.json.\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));fig.subplots_adjust(bottom=.25,wspace=.3)
    for ax,(ds,g)in zip(axs,r['groups'].items()):
        bins=list(g['input_IoU_bins']);vals=[g['input_IoU_bins'][b]['conditional_score']['quantiles']['median']for b in bins]
        ax.bar(range(len(bins)),vals,color='#487b8b');ax.axhline(.35,color='#b75843',ls='--',label='unchanged gate .35');ax.set_xticks(range(len(bins)),bins,rotation=25);ax.set_xlabel('Input overlap with matched native FINAL box');ax.set_ylabel('Conditional score median');ax.set_title(ds+' original development');ax.legend(fontsize=7)
    fig.text(.07,.025,'Cached outputs only. Native final box is a geometry proxy, not GT or native initial reference.\nNo new decoder/GT run. Individual layer paths were not saved and remain unknown.',fontsize=8)
    review=D/'root_review';review.mkdir(exist_ok=True);fig.savefig(P/'REFERENCE_SCORE.svg');fig.savefig(review/'REFERENCE_SCORE.png',dpi=130);plt.close(fig)
    own=['scripts/diagnose_stvg_dino_reference_path_cache32_v1.py','scripts/audit_stvg_dino_reference_path_cache32_v1.py','scripts/report_stvg_dino_reference_path_cache32_v1.py','scripts/publish_stvg_dino_reference_path_cache32_v1.py','protocols/stvg_dino_reference_path_cache32_v1.md']
    files=['DIAGNOSIS.json','CACHE_AVAILABILITY.json','WITHIN_FIXED_CONTENT_MASK.json','ROOT_INDEPENDENT_AUDIT.json','CODE_PATH_AND_LIMITS.json','CONFIGURATION.json','REPORT.md','README.md','REFERENCE_SCORE.svg']
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=own+[str((P/f).relative_to(ROOT))for f in files]));print(dict(status='report_ready',files=len(own)+len(files)))


if __name__=='__main__':main()
