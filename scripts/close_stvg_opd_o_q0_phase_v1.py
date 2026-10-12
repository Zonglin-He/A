"""Close only actually completed O and corrected-Native Q0 authorizations."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha

O = ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
Q = ROOT/'artifacts/stvg_opd_query_observability_q0_v1/Native_support_scope003'


def run():
    assert read(O/'CPU_DIAGNOSTIC_FINAL_COMPLETION.json')['status'] == 'complete'
    for base, root_name, visual_name, completion_name, cells, fits in [
        (O, 'postseal_independent_root001/ROOT_READBACK.json',
         'postseal_independent_root001/ROOT_VISUAL_REVIEW.json', 'COMMON_STATE_CPU_COMPLETION.json', 1792, 5376),
        (Q, 'ROOT_STATISTICS_READBACK.json', 'ROOT_VISUAL_REVIEW.json', 'CPU_COMPLETION.json', 64, 64),
    ]:
        assert read(base/'GLOBAL_PREDICTION_BARRIER.json')['status'] == 'sealed'
        assert read(base/root_name)['status'] == 'pass'
        assert read(base/visual_name)['status'] == 'pass'
        assert read(base/visual_name)['PDF_count'] == 2
        assert read(base/completion_name)['cells'] == cells
        assert read(base/'publication_phase001/PUBLIC_SCALAR_AUDIT.json')['status'] == 'pass'
        remote = read(base/'publication_phase001/FINAL_GITHUB_RECEIPT.json')
        assert remote['status'] == 'pass' and remote['actual_remote_bytes_SHA256_Git_blob_tree_verified']
        assert remote['file_count'] == 125 and remote['bytes'] == 2428807
        for name in ['ARCHIVE_PRE_CLOSING_CHECK', 'ARCHIVE_CLOSING_SNAPSHOT', 'ARCHIVE_CLOSING_CHECK']:
            assert read(base/'publication_phase001'/f'{name}.json')['status'] == 'pass'
        if base == O:
            scope = 'entire authorized O: finite cached CPU geometry diagnostic plus all5376 common-state fits, source/math/state/dense/statistics, two actual PDF pixel views, complete anonymous positive/negative verified publication and archive'
            runtime_paths = [O/'GPU_RUNTIME.json', O/'postseal_independent_root001/RUNTIME.json',
                             O/'postseal_independent_root001/source_origin_stream003/REVISION_RUNTIME.json']
        else:
            scope = 'entire corrected Native Q0 request: all64 frozen diagnostics, exact qualification/repeats/source/state/dense/reward/true-mean/statistics, two actual PDF pixel views, complete anonymous positive/negative verified publication and archive'
            runtime_paths = [Q/'RUNTIME.json', Q/'frozen_memory_revision004/REVISION_RUNTIME.json',
                             Q/'postseal_actual_root001/receipt_binding005/REVISION_RUNTIME.json']
        for runtime_path in runtime_paths:
            for path, digest in read(runtime_path)['pins'].items():
                assert sha(ROOT/path) == digest, path
        value = dict(status='complete', scope=scope, cells=cells, complete_deploy_fits_or_episodes=fits,
                     actual_global_seal=True, original_postseal_scoring_complete=True,
                     actual_independent_root_complete=True, actual_PDF_pixel_views=2,
                     anonymous_all_positive_and_negative_results_complete=True,
                     actual_remote_bytes_SHA256_Git_blob_tree_verified=True,
                     archive_check_snapshot_check_complete=True,
                     global_barrier_sha256=sha(base/'GLOBAL_PREDICTION_BARRIER.json'),
                     independent_root_sha256=sha(base/root_name),
                     actual_visual_review_sha256=sha(base/visual_name),
                     public_scalar_audit_sha256=sha(base/'publication_phase001/PUBLIC_SCALAR_AUDIT.json'),
                     public_remote_receipt_sha256=sha(base/'publication_phase001/FINAL_GITHUB_RECEIPT.json'),
                     archive_check_sha256=sha(base/'publication_phase001/ARCHIVE_CLOSING_CHECK.json'),
                     archive_snapshot_sha256=sha(base/'publication_phase001/ARCHIVE_CLOSING_SNAPSHOT.json'),
                     root_closer_source_sha256=sha(Path(__file__)),
                     original_main_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
                     scientific_configuration_changed=False, new_method_promoted=False,
                     future_OOM_safety_claim=False, S_C_N_complete=False, all_tasks_complete=False,
                     all_five_namespace_monitor_must_remain_ACTIVE=True, CPU_only=True, GT_read=True,
                     time=time.time())
        write(base/'ROOT_CLOSING.json', value)
        write(base/'FINAL_COMPLETION.json', dict(value, ROOT_CLOSING_sha256=sha(base/'ROOT_CLOSING.json'),
                                               task_complete=True))
        status(base/'STATUS.json', dict(status='complete', scope=scope, task_complete=True,
               FINAL_COMPLETION_sha256=sha(base/'FINAL_COMPLETION.json'), CPU_only=True, GT_read=True,
               S_C_N_complete=False, all_tasks_complete=False, time=time.time()))
        print('ACTUAL_PHASE_FINAL', 'O' if base == O else 'correct-Native Q0', 'complete', flush=True)
    write(Q.parent/'FINAL_SCOPE_DISPATCH.json', dict(status='pass', scope='only corrected Native support qualifies for current Q0 final; old full-video scope remains historical and ineligible',
          current_scope='Native_support_scope003', current_FINAL_COMPLETION_sha256=sha(Q/'FINAL_COMPLETION.json'),
          correct_Native_Q0_complete=True, old_scope_qualification_reused=False,
          all_tasks_complete=False, monitor_must_remain_ACTIVE=True, time=time.time()))


if __name__ == '__main__':
    run()
