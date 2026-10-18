"""Cached-native confidence recovery controls; no GT/model access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.diagnose_stvg_dino_reference_path_cache32_v1 import D,P,verify,read_guard,summary,export


def main():
    verify();read_guard();data=load(D/'SCALAR_ROWS_PRIVATE.pt');groups={}
    for ds in ['hc2','vidstg']:
        rows=[r for r in data['rows']if r['dataset']==ds];strong=[r for r in rows if r['native_high_score']>=.35]
        near=[r for r in strong if r['input_native_iou']>=.75];recovered=[r for r in strong if r['output_native_iou']>=.9]
        examples=sorted([r for r in recovered if not r['accepted']],key=lambda r:r['output_native_iou'],reverse=True)[:3]
        groups[ds]=dict(native_score_ge035_and_output_IoU_ge09=summary(recovered),
            native_score_ge035_and_input_IoU_ge075=summary(near),
            same_content_as_mask=summary([r for r in rows if r['content_slot']==r['mask_slot']]),
            different_content_from_mask=summary([r for r in rows if r['content_slot']!=r['mask_slot']]),
            recovered_same_content_as_mask=summary([r for r in recovered if r['content_slot']==r['mask_slot']]),
            recovered_different_content_from_mask=summary([r for r in recovered if r['content_slot']!=r['mask_slot']]),
            small_center_and_size_error=summary([r for r in strong if r['input_center_relative']<=.25 and r['input_log_size']<=.25]),
            counterexamples=[{k:r[k]for k in ['alias','dataset','position','sample','content_slot','mask_slot','score','native_high_score','input_native_iou','output_native_iou']}for r in examples])
    export(P/'NATIVE_CONFIDENCE_RECOVERY_CONTROLS.json',dict(status='cached_only_complete',groups=groups,
        descriptive_controls_not_new_deployment_thresholds=True,
        mask_slot_covaries_with_reference_nearest_native_proposal=True,
        mask_equals_content_not_separate_causal_effect=True,missing_same_reference_alternate_mask_counterfactual=True,
        no_new_GT_model_or_media=True))
    lines=['# Confidence recovery controls and attention-context confounding','','Restrict to native target score>=.35, then inspect conditional final overlap>=.9 with the matched native final candidate geometry. This avoids calling a naturally subthreshold native query a confidence-collapse case. These are geometric proxy controls, not GT or a new gate.','','| Domain | High native score, recovered geometry >=.9 | Admitted | Rejected |','|---|---:|---:|---:|']
    for ds,g in groups.items():
        x=g['native_score_ge035_and_output_IoU_ge09'];lines.append(f'| {ds} | {x["references"]} | {x["admitted"]} | {x["references"]-x["admitted"]} |')
    lines+=['','All existing admitted feedback uses the same mask slot as high content. Changing student reference also changes the nearest native slot and hence the original exclusion mask across samples. The within-frame/content/mask analysis controls this for its comparisons. Cross-mask statistics are descriptive: they cannot prove mask effects, and it is not valid to attribute every confidence difference purely to physical distance. A matched same-reference alternate-mask response is not cached.','','The highest-overlap rejected examples in the JSON are selected by this disclosed deterministic rule, without GT, to demonstrate that final geometric recovery is not sufficient for score recovery. They do not establish event/identity correctness, and repeated candidates share the same query. No new layer/reference/mask experiment was run.']
    (P/'CONFIDENCE_RECOVERY_INTERPRETATION.md').write_text('\n'.join(lines)+'\n')
    print({ds:dict(recovered=g['native_score_ge035_and_output_IoU_ge09']['references'],admitted=g['native_score_ge035_and_output_IoU_ge09']['admitted'],examples=g['counterexamples'])for ds,g in groups.items()})


if __name__=='__main__':main()
