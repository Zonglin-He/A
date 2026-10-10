"""Actual final experimental-suite closure, only after all phase/public/archive requirements."""
import json
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import verify,BASE,NS,read,write,sha,status


def run():
    verify();a=read(BASE/'PAPER_SUITE_ROOT_ASSEMBLY.json');g=read(BASE/'PAPER_SUITE_FINAL_GITHUB_RECEIPT.json')
    ar=read(BASE/'PAPER_SUITE_ARCHIVE_VERIFICATION_RECEIPT.json');portable=read(BASE/'PAPER_SUITE_PUBLIC_AUDIT.json')
    assert a['phase_count']==6 and g['status']==ar['status']==portable['status']=='pass'
    assert g['actual_remote_bytes_SHA256_Git_blob_tree_verified'] and set(g['remote_refs_after'].values())=={g['commit']}
    assert ar['history_sha256']==sha(ROOT/'docs/RESEARCH_HISTORY.md')
    assert portable['phase_count']==6 and portable['comparisons']==3420
    phase_receipts={}
    for item in a['phases']:
        p=BASE/(item['phase']+'_ROOT_CLOSING_RECEIPT.json');c=read(p)
        assert c['status']=='complete' and sha(p)==item['root_closing_sha256'] and c['all_negative_results_retained']
        phase_receipts[item['phase']]=sha(p)
    for p,h in a['outputs'].items():assert sha(ROOT/p)==h,p
    evidence=[BASE/'PAPER_SUITE_ROOT_ASSEMBLY.json',BASE/'PAPER_SUITE_PUBLIC_AUDIT.json',BASE/'PAPER_SUITE_FINAL_GITHUB_RECEIPT.json',BASE/'PAPER_SUITE_ARCHIVE_VERIFICATION_RECEIPT.json',ROOT/'docs/STVG_OPD_REVISED_SUITE_ROOT_REVIEW.md']
    close=dict(status='complete',scope='original explicitly authorized fixed paper experimental suite P1-P6 actual GPU CPU root real-view all-negative anonymous-publication and archive closure',
        paper_suite_complete=True,original_authorized_phases=['P1','P2','P3','P4','P5','P6'],actual_phase_root_sha256=phase_receipts,
        actual_phase_count=6,all_stage_global_prediction_seals_before_GT=True,all_complete_saved_math_state_dense_root_reviews=True,
        all_original_actual_report_RGB_views_complete=True,all_negative_results_preserved=True,all_original_phase_public_bytes_SHA256_Gitblob_tree_verified=True,
        actual_final_suite_public_bytes_SHA256_Gitblob_tree_verified=True,actual_final_suite_public_commit=g['commit'],
        actual_final_suite_public_files=g['file_count'],actual_final_suite_public_bytes=g['bytes'],portable_contract_binding_comparisons=portable['comparisons'],
        current_method_unchanged=True,EATA_and_all_historical_queues_remain_paused=True,no_result_driven_retuning_or_roster_change=True,
        all_original_experimental_phases_have_no_remaining_GPU_CPU_root_view_publication_work=True,
        independent_full_decoder_Jacobian=False,future_numerical_OOM_guarantee=False,whole_manuscript_or_acceptance_claim=False,
        final_closing_receipt_publication_and_postclosing_archive_required=True,
        evidence={str(p.relative_to(ROOT)):sha(p) for p in evidence},time=time.time())
    write(BASE/'PAPER_SUITE_ROOT_CLOSING_RECEIPT.json',close)
    final=dict(status='complete',scope='actual complete original fixed P1-P6 experimental suite',paper_suite_complete=True,
        phases=['P1','P2','P3','P4','P5','P6'],P1_P6_actual_root_complete=True,actual_all_stage_prediction_seal_and_postseal_GT=True,
        actual_full_saved_math_state_dense=True,actual_report_and_RGB_view=True,actual_anonymous_all_negative_public_content_verified=True,
        actual_archive_check_snapshot_check=True,whole_manuscript_or_acceptance_claim=False,main_method_changed=False,
        EATA_and_all_historical_queues_remain_paused=True,next_unauthorized_experiments_started=False,
        actual_root_closing_sha256=sha(BASE/'PAPER_SUITE_ROOT_CLOSING_RECEIPT.json'),actual_public_commit=g['commit'],
        phase_root_sha256=phase_receipts,evidence=close['evidence'],time=time.time())
    write(BASE/'FINAL_COMPLETION.json',final)
    prior=BASE/'PAPER_SUITE_ROOT/prior_dynamic';prior.mkdir(parents=True,exist_ok=True)
    for name in ['STATUS.json','STAGE.json','COMPONENT_STAGE.json','LATER_CPU_STAGE.json']:
        p=BASE/name
        if p.exists():shutil.copy2(p,prior/name)
        status(p,dict(status='complete_original_fixed_P1_P6_suite',stage='FINAL',paper_suite_complete=True,
            final_completion_sha256=sha(BASE/'FINAL_COMPLETION.json'),root_closing_sha256=sha(BASE/'PAPER_SUITE_ROOT_CLOSING_RECEIPT.json'),
            EATA_and_historical_queues_paused=True,final_receipt_publication_in_progress=True))
    status(NS/'STATUS.json',dict(status='P6_and_original_fixed_paper_suite_complete',P6_phase_complete=True,paper_suite_complete=True,final_completion_sha256=sha(BASE/'FINAL_COMPLETION.json')))
    print(json.dumps(dict(status='original_P1_P6_actual_complete',FINAL_COMPLETION_sha256=sha(BASE/'FINAL_COMPLETION.json'),suite_root_sha256=sha(BASE/'PAPER_SUITE_ROOT_CLOSING_RECEIPT.json'),final_metadata_publication_and_postclosing_archive_pending=True)))


if __name__=='__main__':run()
