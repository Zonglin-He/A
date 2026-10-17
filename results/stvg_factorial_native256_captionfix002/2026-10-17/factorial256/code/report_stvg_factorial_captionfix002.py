"""Preserve old report source, disclose the repaired Caption-matched factorial."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import BASE as REPAIR,NATIVE,FIXED,read,verify
def main():
    from scripts import report_stvg_factorial_native256_v1 as r,audit_stvg_factorial_readout_v1 as audit
    audit.BASE=FIXED;audit.verify=verify
    src=inspect.getsource(r.main).replace('stvg_factorial_native256_v1/2026-10-17/factorial256','stvg_factorial_native256_captionfix002/2026-10-17/factorial256').replace('STVG_NATIVE_FACTORIAL256_001.md','STVG_NATIVE_FACTORIAL_CAPTIONFIX002.md')
    src=src.replace('Existing Frozen, Temporal and Adapted-clock Full outputs are reused exactly.','Frozen, Temporal and Full-A are reused exactly. Full-B uses the old221 complete outputs plus35 actual Native-B full outputs only after exact same-clock E1 jobs, Source, Temporal interval and recipe checks. Historical original Full-B remains a separate row; the new comparison explicitly repairs Caption acquisition on both clocks.')
    # Dispatch receipts live in the repair controller, not the scientific scope.
    assert src.count("(BASE/'execution').glob")==1
    src=src.replace("(BASE/'execution').glob","(REPAIR/'execution').glob")
    src=src.replace('Source_and_Temporal_and_Adapted_Full_are_existing_sealed_reuses_not_new_model_work=True','Frozen_Temporal_and_Full_A_reused=True,Full_B_original221_reused_and35_actual_Native_B_outputs_reused_only_after_exact_binding=True,Caption54_and_extra2_Caption2_alignment_in_shared_generate_stage=True')
    src=src.replace('feature_build=read(BASE/\'FEATURE_BARRIER.json\'),','feature_build=read(BASE/\'FEATURE_BARRIER.json\'),initial_Native_feature_build=initial_feature_scalars,initial_feature_stage_total_wall_seconds=None,initial_feature_stage_total_wall_status="unknown; original parent retired after normal child exit",')
    src=src.replace("save('RESULTS.json',d);","save('RESULTS.json',d);save('ROOT_INDEPENDENT_SCALAR_READBACK003.json',read(BASE/'ROOT_INDEPENDENT_SCALAR_READBACK003.json'));save('ROOT_ACTUAL_PIXEL_VIEW_READBACK.json',read(BASE/'ROOT_ACTUAL_PIXEL_VIEW_READBACK.json'));")
    # Existing historical B must be visible in the table as well as raw JSON.
    src=src.replace("'Full_A_Adapted','Full_B_Adapted'):","'Full_A_Adapted','Full_B_Adapted','Full_B_original_missingCaption'):")
    src=src.replace('Native-candidate Caption is reused without new Temporal-parameter updates.','Existing available Native-candidate Captions are reused and54 previously unrequested valid clips are generated separately (35 core and19 confirmation; plus2 Caption/2 alignment qualification calls), without new Temporal-parameter updates. For43 parents with identical Native and Adapted intervals, exact original E1 candidate pools and marked pixels are reused; other213 use the prepared Native inputs. Original Native feature work completes before handoff and is reused; corrected feature preparation adds only16 visual encodes and reuses1008 existing visual features. Initial feature-stage total wall time was not captured by the stopped parent and remains unknown; its recorded encoder/model-load counters are preserved separately. Shared generation and all new controller stages are included once in the cost table.')
    src=src.replace("'publish_stvg_factorial_native256_v1.py'):","'publish_stvg_factorial_native256_v1.py','stvg_spatial_caption_repair_v1.py','continue_stvg_captionfix_native002.py','score_stvg_factorial_captionfix002.py','report_stvg_factorial_captionfix002.py','publish_stvg_factorial_captionfix002.py','audit_stvg_factorial_scalars003.py'):")
    initial={k:v for k,v in read(NATIVE/'FEATURE_BARRIER.json').items() if isinstance(v,(int,float,bool)) or k=='status'}
    ns=dict(r.__dict__);ns.update(BASE=FIXED,REPAIR=REPAIR,initial_feature_scalars=initial,verify=verify);exec(compile(src,__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()
