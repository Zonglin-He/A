"""Close only the fully verified finite P9 cached actionability audit."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE,OUT,read,write,sha,verify
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
    verify();view=read(BASE/'ROOT_VISUAL_REVIEW.json');public=read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert all(read(BASE/n)['status']=='pass' for n in ['ROOT_READBACK.json','ROOT_EXTERNAL_SIGNAL_READBACK.json'])
    assert view['status']==public['status']=='pass' and view['actual_pixel_view'] and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    for relative,b in view['PDF_files'].items():assert sha(ROOT/relative)==b['sha256'] and (ROOT/relative).stat().st_size==b['bytes']
    assert not read(BASE/'CANDIDATE_DECISION.json')['heldout_signal_supports_bounded_counterfactual_selection_test']
    note=f"**2026-10-13｜P9缓存空间专家证据可行动性审计全部实际完成。** 原每源128历史parent/两order512cell、1024旧Omega0/Omega1更新，全部512新无GT读出先封存后GT标签，0新GPU/model/DINO/optimizer；完整独立root40241scalar、513096坐标、2048原件二次SHA和独立无GT信号10240scalar通过，真实两PDF五页pixel view与全部正负/中性/unknown匿名公开齐全。教师平均IoU≥.5的HC两观测支持仍22/64与23/63有害，Vid15/60与15/62有害；严格全部已标注anchor正确的有害病例也保留。heldout四组AUROC区间均跨.5，在相同eligible parents对detector score的配对差区间均跨0，未达到预锁counterfactual筛选条件，不加gate/阈值/新方法，不能概括全部无标签信号均无信息。{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际pass。P9符号翻转仅初步关联，不证明LN继承因果；人类已授权随后另存P9-P0同证据/随机数/跨顺序状态分析，C expanded GPU继续原任务，剩余baseline保持明确取消，未写论文。见[实际P9root](</home/wwww/visual grounding/artifacts/stvg_opd_evidence_actionability_audit_v1/ROOT_READBACK.json>)、[完整匿名结果](</home/wwww/visual grounding/results/stvg_opd_evidence_actionability_audit/2026-10-12/REPORT.md>)。\n\n"
    p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
    for a in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(a)==1;text=text.replace(a,a+note,1)
    p.write_text(text);archive(BASE/'archive_actual_P9_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings={n:sha(BASE/n) for n in ['ROOT_READBACK.json','ROOT_EXTERNAL_SIGNAL_READBACK.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json','GLOBAL_READOUT_BARRIER.json','CANDIDATE_DECISION.json']}
    scope='complete finite P9 cache-only teacher geometry versus update consequences and heldout evidence predictability audit; subsequent P9-P0 and C/N/cost remain separate authorized work'
    write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,cells=512,saved_updates=1024,parents=256,PDFs=2,pages=5,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=True,time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,new_gate_created=False,causal_LN_claim=False,whole_followup_request_complete=False,production_promoted=False,CPU_only=True,GT_read=True,time=time.time()))
    print('P9_FINITE_AUDIT_ACTUAL_CLOSED',public['commit'])
if __name__=='__main__':run()
