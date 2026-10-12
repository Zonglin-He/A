"""Close the actual bounded cached analysis after root/view/public/archive."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.analyze_stvg_opd_temporal_spatial_v1 import verify,BASE

def run():
    verify();runtime=read(BASE/'RUNTIME.json')
    roots=['ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']
    for f in roots:assert read(BASE/f)['status']=='pass',f
    assert read(BASE/'GLOBAL_READOUT_BARRIER.json')['logical_outputs']==1536
    assert read(BASE/'Q0_CANCELLATION_READBACK.json')['pairs']==1308
    public=read(BASE/'FINAL_GITHUB_RECEIPT.json');assert public['actual_remote_bytes_SHA256_Git_blob_tree_verified'] and public['file_count']==33
    source=read(BASE/'PAPER_REVISION_RECEIPT.json')
    for f,h in source['source_files'].items():assert sha(ROOT/f)==h
    scope='bounded original64-parent P6 factorial/exact O128 two-order readout/Q0 saved64 mean diagnosis/full negative scalar publication/two actual viewed PDFs/archive/working paper sources'
    evidence={f:sha(BASE/f) for f in roots+['GLOBAL_READOUT_BARRIER.json','Q0_CANCELLATION_READBACK.json','PAPER_REVISION_RECEIPT.json','PAPER_PREVIEW_COMPILER_RECEIPT.json']}
    write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,evidence=evidence,CPU_only=True,new_GPU_fits=0,new_model_calls=0,
        actual_PDF_pixel_view=True,anonymous_all_positive_negative_results_public=True,actual_remote_bytes_SHA256_Git_blob_tree_verified=True,
        full_1792_O_extension_completed=False,unavailable_O_cells=1664,missing_time_fields_declared=True,
        original_Full_changed=False,current_method_promoted=False,new_Query_OPD=False,temperature_changed=False,
        standalone_LaTeX_compile_success=False,builtin_compiler_bundle_unavailable=True,paper_source_revision_complete=True,
        S_C_N_complete=False,all_current_authorized_tasks_complete=False,public_science_commit=public['commit'],source_sha256=sha(Path(__file__)),time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        root_view_public_archive_actual=True,full_1792_extension_completed=False,unavailable_O_cells=1664,
        paper_working_source_revision_complete=True,paper_PDF_preview_unverified=True,builtin_compiler_bundle_unavailable=True,
        new_GPU_fits=0,new_model_calls=0,original_S_C_N_continue=True,all_supplements_complete=False,
        half_hourly_monitor_must_remain_ACTIVE=True,time=time.time()))
    print('BOUNDED_CACHED_T_FINAL_COMPLETE')

if __name__=='__main__':run()
