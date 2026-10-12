"""Close only genuine existing S evidence; future main and cancellation stay explicit."""
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
from scripts.close_stvg_opd_p7_phase_v1 import archive
BASE=ROOT/'artifacts/stvg_opd_supplement_v1'

def run():
    for n in ['ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json']:
        assert read(BASE/n)['status']=='pass'
    public=read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert public['status']=='pass' and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    view=read(BASE/'ROOT_VISUAL_REVIEW.json');assert view['actual_pixel_view'] and view['PDFs']==6 and view['pages']==11
    for n,b in view['PDF_files'].items():assert sha(ROOT/n)==b['sha256'] and (ROOT/n).stat().st_size==b['bytes']
    decision=read(BASE/'FOLLOWUP_EVIDENCE_DECISION.json')
    assert decision['existing_evidence_decision_complete'] and not decision['future_method_and_main_protocol_complete']
    note=f"**2026-10-13｜S取消剩余formal后的当前证据scope实际闭合。** 原2304顺序完整source/math/state/input/dense与原独立root/后N同2304完整复核、全部1152开发结果/18配置和原锁选参、1244保存观察/未观察scalar及237parent表格34.229复算、原32实际cost、六PDF十一页原真实pixel view沿逐byte SHA精确复用均齐全；34639已接受formal原件（两HC各10446及Vid13747prefix）与1152开发原件逐bytes/SHA/receipt重新核验，不反序列化、不评分取消formal。{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际pass；全正负neutralunknown匿名公开，六页私人RGB/GT固定案例仅绑定匿名provenance，不公开图像。原baseline同PID/start/optimizer/ordinarylease T standby及原publisher/root006科学锁保持，另存publication-only接续；0新GPU/model/DINO/optimizer/rawGT评分/论文/CURRENT晋升。S当前证据真实FINAL不等于82710 formal完成或未来main完成。现有证据裁决已记录：P7固定耦合扩大条件未触发/P9 heldout gate未建立/P0自然关联非LN因果/C reset双域优势未建立/N组件效应有域差；未来方法和主实验协议仍单独pending，半小时monitor ACTIVE，未经新明确授权不自动新gate/因果state/teacher/完整main或恢复cancelled baseline。见[完整S报告](</home/wwww/visual grounding/results/stvg_opd_supplement/2026-10-13/after_cancel_complete/REPORT.md>)。\n\n"
    ledger=ROOT/'docs/RESEARCH_HISTORY.md';text=ledger.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n','## 7. 候选路线总登记与重开条件\n\n']:
        assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
    ledger.write_text(text)
    archive(BASE/'archive_after_cancel_closing001','ARCHIVE_VERIFICATION_RECEIPT.json')
    scope='all existing authorized S evidence after explicit remaining-formal cancellation: orders2304/dev1152/custody34639/saved effects/fixed cases/cost32/root/actual prior11 PDF pages/all anonymous public/archive; not82710 formal or future main'
    names=['ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json','PUBLIC_SCALAR_AUDIT.json','FOLLOWUP_EVIDENCE_DECISION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']
    common=dict(status='complete',scope=scope,actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,
        current_authorized_S_evidence_complete=True,original_formal_baseline_evaluation_complete=False,baseline_remaining_formal_cancelled=True,
        existing_evidence_decision_complete=True,future_method_and_main_protocol_complete=False,whole_followup_request_complete=False,
        production_promoted=False,manuscript_written=False,new_GPU_calls=0,new_model_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=False,time=time.time())
    write(BASE/'ROOT_CLOSING.json',dict(common,bindings={n:sha(BASE/n) for n in names},PDFs=6,pages=11))
    write(BASE/'FINAL_COMPLETION.json',dict(common,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),all_negative_neutral_unknown_public=True,
        accepted_formal_artifacts_retained=34639,Vid_TENT_prefix_retained=13747,development_arrivals=1152,order_fits=2304,cost_actual_paths=32))
    status(BASE/'STATUS.json',dict(status='complete_current_S_evidence_future_main_protocol_pending',scope=scope,CPU_only=True,GT_read=False,time=time.time()))
    print('S_CURRENT_EXISTING_AUTHORIZED_EVIDENCE_GENUINE_FINAL',public['commit'])

if __name__=='__main__':run()
