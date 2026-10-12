"""Actual Expanded-State closure after root, anonymous audit, view and remote bytes."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.stvg_opd_expanded_state_common_v1 import BASE,verify
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
    verify();root=read(BASE/'ROOT_COMPLETE_READBACK.json');view=read(BASE/'ROOT_VISUAL_REVIEW.json');public=read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert root['status']==view['status']==public['status']=='pass' and root['new_formal_fits']==5430
    assert view['actual_pixel_view'] and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    assert read(BASE/'PUBLIC_SCALAR_AUDIT.json')['status']=='pass'
    for rel,b in view['PDF_files'].items():assert sha(ROOT/rel)==b['sha256'] and (ROOT/rel).stat().st_size==b['bytes']
    note=f"**2026-10-13｜C Expanded-State 5430正式拟合及完整有限结果实际关闭。** 原205HC＋700Vid历史非开发parent/两固定order/Full-alpha0-alpha1全部5430新fit封存后GT；原9730560状态坐标、91200轮math-state-source-input/dense、6335二次bytesSHA、24实际qualified-formal/3918102tensor坐标、33114独立paired endpoint与全部905parent匿名scalar/bootstrap/负尾复算真实pass。Full−alpha0总体HC+0.066pp/CI[-0.171,+0.305]、Vid+0.237/[-0.015,+0.491]均跨零；Full−alpha1 HC+0.337/[-0.094,+0.790]仍跨零，Vid+0.809/[+0.570,+1.055]及两个order均正。仅Vid衰减相对完全继承收益建立，未证明两域稳定优于reset；Before−Frozen不可代Full−reset，也未证明逐queryLN因果/无标签gate。全部5430匿名正负neutralunknown/905配对、所有绝对与current-update严重负尾/原真实PDF一页pixel view公开，{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际通过。原GPU正常退出、0新增DINO/本次export无新GPU或GT评分，原scientific source/runtime未改；未写论文或晋升CURRENT。仅Expanded有限scope完成，原128真实部署计时/LN768/32cost及后续决定仍须新实际串行接续，剩余formal baseline维持人类cancel13747standby，不能要求或伪造82710global。见[完整C结果](</home/wwww/visual grounding/results/stvg_opd_claim_strengthening/2026-10-13/expanded_state_complete/REPORT.md>)。\n\n"
    p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n','## 7. 候选路线总登记与重开条件\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(anchor)==1;text=text.replace(anchor,anchor+note,1)
    p.write_text(text);archive(BASE/'archive_expanded_actual_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings={n:sha(BASE/n) for n in ['GLOBAL_PREDICTION_BARRIER.json','CPU_SCORING_COMPLETION.json','ROOT_COMPLETE_READBACK.json',
        'ROOT_ANONYMOUS_SCALAR_READBACK.json','PUBLIC_SCALAR_AUDIT.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']}
    scope='complete authorized finite C Expanded-State 905 historical non-development parents/two orders/three rules/all5430 new formal fits/full independent root/actual PDF pixel view/all anonymous outcomes/public/archive; pending deployment timing/LN/cost'
    write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,new_formal_fits=5430,parents=905,
        PDFs=1,pages=1,new_DINO_calls=0,actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,
        complete_current_phase=True,whole_claim_strengthening_complete=False,whole_followup_request_complete=False,
        CPU_only=True,GT_read=True,time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,
        deployment_timing128_complete=False,LN768_complete=False,cost32_complete=False,whole_claim_strengthening_complete=False,
        whole_followup_request_complete=False,new_gate_created=False,new_state_intervention_launched=False,
        production_promoted=False,CPU_only=True,GT_read=True,time=time.time()))
    print('C_EXPANDED_ACTUAL_FINITE_SCOPE_CLOSED',public['commit'])
if __name__=='__main__':run()
