"""Genuine full finite P9-P0 cache-analysis closure, no state intervention."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_conditional_actionability_p0_v1 import BASE,OUT,read,write,sha,verify
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
    verify();view=read(BASE/'ROOT_VISUAL_REVIEW.json');public=read(BASE/'FINAL_GITHUB_RECEIPT.json');root=read(BASE/'ROOT_READBACK.json')
    assert view['status']==public['status']==root['status']=='pass' and view['actual_pixel_view'] and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    for relative,b in view['PDF_files'].items():assert sha(ROOT/relative)==b['sha256'] and (ROOT/relative).stat().st_size==b['bytes']
    note=f"**2026-10-13｜P9-P0同专家证据/固定随机数、跨顺序状态机制分析全部实际完成。** 每源128历史parent/两order512cell、512parent-scope配对/1024保存更新，全部query/source/pixel/物理帧/expert/admission/config精确匹配、query256均0、LN1536状态差异；916实际有采样episode/24160轮/11128320sample坐标二次精确CPU replay，108空证据原no-information无sampler/optimizer完整保留。全917504初态坐标与256缓存先封存再复用P9原GT离线标签，0新GPU/model/DINO/optimizer/rawGT评分；2048原来源二次SHA、1382012独立scalar/root和真实两PDF四页pixel view全部pass。common-U跨order符号翻转HC Omega0/1为25/18（各128），Vid为14/14；两侧至少1pp为14/9/6/10，两侧至少5pp仅0/1/0/2，不能把全部符号翻转当严重损害反转。绝对效果差均值3.803/3.120/3.916/3.943pp；HC部分LN距离关联区间正，Vid尚未稳定，全部旧32/新96/128、own-U、neutral/unknown和71条parent-scope翻转完整保留。自然state/order关联不证明LN因果或无标签gate有效，待原C5430全global/root后才决定受控状态交叉干预；原GPU未打断，剩余baseline维持人类cancel，未写论文。{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际pass。原69接受parent/空证据读取dispatch全部保存解释且science runtime不改。见[完整P0报告](</home/wwww/visual grounding/results/stvg_opd_state_conditional_actionability/2026-10-13/REPORT.md>)、[真实独立root](</home/wwww/visual grounding/artifacts/stvg_opd_state_conditional_actionability_v1/ROOT_READBACK.json>)。\n\n"
    p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
    for a in ['## 1. 当前状态：先读这一节\n\n','## 7. 候选路线总登记与重开条件\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(a)==1;text=text.replace(a,a+note,1)
    p.write_text(text);archive(BASE/'archive_actual_P9P0_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings={n:sha(BASE/n) for n in ['ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','GLOBAL_PAIRED_READOUT_BARRIER.json','MECHANISM_SCOPE_DECISION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json','empty_episode_readback001/RESOLUTION_RECEIPT.json']}
    scope='complete authorized finite P9-P0 same evidence/common random numbers/order-dependent update effects and inherited LN state association analysis; complete C outcome required before any later controlled state intervention'
    write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,parents=256,parent_scope_pairs=512,saved_updates=1024,PDFs=2,pages=4,actual_raw_GT_read=False,new_GT_scoring_calls=0,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=True,time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,causal_LN_attribution=False,new_gate_created=False,new_state_intervention_launched=False,C_complete=False,whole_followup_request_complete=False,production_promoted=False,CPU_only=True,GT_read=True,time=time.time()))
    print('P9_P0_ACTUAL_FINITE_SCOPE_CLOSED',public['commit'])
if __name__=='__main__':run()
