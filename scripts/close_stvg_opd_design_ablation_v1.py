"""Close only the complete, viewed and remotely verified component experiment.

This writes additive receipts. It performs no inference, scoring, GT access,
source-state update, scheduling or GitHub mutation. Archive maintenance and the
second metadata publication remain explicit root operations.
"""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import BASE, PUB, read, write, sha, verify


def evidence():
    verify()
    barrier = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    root = read(BASE / 'actual_root/ROOT_ALL_FITS_READBACK.json')
    visual = read(BASE / 'ROOT_VISUAL_REVIEW.json')
    cases = read(BASE / 'actual_fixed_case_views002/COMPLETION.json')
    scalar = read(BASE / 'PUBLIC_SCALAR_AUDIT.json')
    interpreted = read(BASE / 'ROOT_INTERPRETATION.json')
    assert barrier['new_formal_fits'] == 5376
    assert root['status'] == visual['status'] == scalar['status'] == 'pass'
    assert root['counts']['complete_fits'] == scalar['new_formal_rows'] == 5376
    assert cases['cases'] == 6
    assert interpreted['all_negative_results_retained']
    assert interpreted['visual_sha256'] == sha(BASE / 'ROOT_VISUAL_REVIEW.json')
    assert interpreted['report_sha256'] == sha(PUB / 'ACTUAL_ROOT_REVIEW.md')
    assert read(BASE / 'CPU_actual_root_continuation003/COMPLETION.json')['status'].startswith('pending_actual_root')
    return root


def close(archive_path):
    root = evidence()
    archive_path = Path(archive_path)
    archive = read(archive_path)
    remote = read(BASE / 'FINAL_GITHUB_RECEIPT.json')
    assert archive['status'] == remote['status'] == 'pass'
    assert [c['mode'] for c in archive['checks']] == ['check', 'snapshot', 'check']
    assert all(c['returncode'] == 0 for c in archive['checks'])
    assert remote['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    result = dict(
        status='complete',
        scope='three fixed matched component ablations, combined historical/new table, offline event-coverage Figure3, fixed historical case verification, actual views and verified public results',
        new_formal_fits=5376,
        qualification_arrivals=72,
        qualification_complete_GPU_fits=144,
        qualified_predictions_used_as_formal=False,
        root_counts=root['counts'],
        all_deployment_arms_sealed_before_new_GT=True,
        all_negative_results_retained=True,
        actual_root_math_state_dense_readback=True,
        actual_report_and_private_RGB_views=True,
        anonymous_public_results_verified=True,
        result_commit=remote['commit'],
        public_file_count=remote['file_count'],
        public_bytes=remote['bytes'],
        root_readback_sha256=sha(BASE / 'actual_root/ROOT_ALL_FITS_READBACK.json'),
        visual_review_sha256=sha(BASE / 'ROOT_VISUAL_REVIEW.json'),
        public_scalar_audit_sha256=sha(BASE / 'PUBLIC_SCALAR_AUDIT.json'),
        result_remote_receipt_sha256=sha(BASE / 'FINAL_GITHUB_RECEIPT.json'),
        archive_receipt_sha256=sha(archive_path),
        scientific_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        historical_original_suite_unchanged=True,
        production_promotion=False,
        new_TubeDETR=False,
        EATA_or_historical_queue_restored=False,
        historical_exposed_cohort=True,
        coverage_is_offline_association_not_causal_proof=True,
        independent_complete_decoder_Jacobian=False,
        CPU_only=True,
        model_calls=0,
        time=time.time(),
    )
    write(BASE / 'ROOT_CLOSING_RECEIPT.json', result)
    print('DESIGN_ABLATION_ROOT_CLOSED', sha(BASE / 'ROOT_CLOSING_RECEIPT.json'), flush=True)


def final(archive_path):
    evidence()
    closing = read(BASE / 'ROOT_CLOSING_RECEIPT.json')
    remote = read(BASE / 'FINAL_METADATA_GITHUB_RECEIPT.json')
    archive_path = Path(archive_path)
    archive = read(archive_path)
    assert closing['status'] == 'complete'
    assert remote['status'] == archive['status'] == 'pass'
    assert remote['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    assert [c['mode'] for c in archive['checks']] == ['check', 'snapshot', 'check']
    assert all(c['returncode'] == 0 for c in archive['checks'])
    assert set(remote['remote_refs_after'].values()) == {remote['commit']}
    result = dict(
        status='complete', scope=closing['scope'], experiment_complete=True,
        new_formal_fits=5376, original_completed_suite_unchanged=True,
        result_commit=closing['result_commit'], final_public_commit=remote['commit'],
        final_remote_file_count=remote['file_count'], final_remote_bytes=remote['bytes'],
        final_remote_receipt_sha256=sha(BASE / 'FINAL_METADATA_GITHUB_RECEIPT.json'),
        root_closing_sha256=sha(BASE / 'ROOT_CLOSING_RECEIPT.json'),
        actual_archive_receipt_sha256=sha(archive_path),
        all_root_view_public_archive_requirements_complete=True,
        no_new_TubeDETR=True, no_retuning_or_production_promotion=True,
        no_recurring_automation_created=True, CPU_only=True, model_calls=0,
        time=time.time(),
    )
    write(BASE / 'FINAL_COMPLETION.json', result)
    print('DESIGN_ABLATION_COMPLETE', remote['commit'], flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] in {'close', 'final'}
    {'close': close, 'final': final}[sys.argv[1]](sys.argv[2])
