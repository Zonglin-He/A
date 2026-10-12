"""Regenerate only figure layout from identical P9 sealed scalar results."""
import os,sys,shutil,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE,OUT,read,write,sha,verify
def run():
    verify();folder=BASE/'figure_layout002';work=folder/'render_root';results=work/'scalar_copy';results.mkdir(parents=True,exist_ok=True)
    for name in ['ROWS.jsonl.gz','SUMMARY.json']:
        target=results/name
        if target.exists():assert sha(target)==sha(OUT/name)
        else:shutil.copy2(OUT/name,target)
    metadata=work/'metadata';metadata.mkdir(exist_ok=True)
    for name in ['ROOT_READBACK.json','CANDIDATE_DECISION.json']:
        target=metadata/name
        if target.exists():assert sha(target)==sha(BASE/name)
        else:shutil.copy2(BASE/name,target)
    from scripts import report_stvg_opd_actionability_audit_v1 as report
    report.OUT=results;report.ROOT=work;report.BASE=work/'metadata'
    if not (report.BASE/'FIGURE_EXPORT.json').exists():report.run()
    for name in ['ROWS.jsonl.gz','SUMMARY.json','REPORT.md','ALL_PARENT_UPDATES.csv','CORRECT_TEACHER_HARMFUL_UPDATES.csv','REPEATED_ORDER_DIAGNOSTICS.json']:
        assert sha(results/name)==sha(OUT/name),name
    exported=read(report.BASE/'FIGURE_EXPORT.json');bindings={}
    for relative in exported['files']:
        source=work/relative;target=ROOT/relative;shutil.copy2(source,target)
        assert sha(source)==sha(target);bindings[relative]=dict(sha256=sha(target),bytes=target.stat().st_size)
    receipt=BASE/'FINAL_FIGURE_EXPORT.json'
    if receipt.exists():assert read(receipt)['files']==bindings
    else:write(receipt,dict(status='pending_actual_final_layout_pixel_view_publication_archive',PDFs=2,pages=5,files=bindings,previous_export_sha256=sha(BASE/'FIGURE_EXPORT.json'),CPU_only=True,GT_read=True,time=time.time()))
    resolution=folder/'RESOLUTION_RECEIPT.json'
    if resolution.exists():assert read(resolution)['status']=='pass' and read(resolution)['all_scalar_outputs_byte_identical'] and read(resolution)['files']==bindings
    else:write(resolution,dict(status='pass',scope='same complete scalar CSV/report/rows/summary bytes; only plot annotations/color-key layout regenerated, final export separate from immutable original',all_scalar_outputs_byte_identical=True,files=bindings,new_GPU_fits=0,new_model_calls=0,new_GT_scoring_calls=0,CPU_only=True,GT_read=False,time=time.time()))
    print('P9_LAYOUT_ONLY_AND_ALL_SCALAR_BYTES_EXACT_PASS')
if __name__=='__main__':run()
