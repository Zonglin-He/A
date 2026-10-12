"""Genuine finite P8 CPU-factorial closure; later main reconstruction remains separate."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_support_factorization_v1 import BASE,OUT,read,write,sha,verify
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
    verify();root=read(BASE/'ROOT_READBACK.json');view=read(BASE/'ROOT_VISUAL_REVIEW.json');public=read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert root['status']==view['status']==public['status']=='pass' and view['actual_pixel_view'] and public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    for name,b in view['PDF_files'].items():assert sha(ROOT/name)==b['sha256'] and (ROOT/name).stat().st_size==b['bytes']
    s=read(OUT/'SUMMARY.json');m={ds:s[ds]['all128']['both']['statistics']['interaction_v'] for ds in ['hc2','vidstg']}
    note=f"**2026-10-12｜P8缓存2×2全部实际CPU/global后评分/独立来源坐标dense与bootstrap/真实两PDF四页view/匿名全正负公开/档案有限关闭。** 原每源128历史parent/两order512cell、2048factorial读出和512新E还原I0，0GPU/model/DINO/optimizer；全部128 interaction HC{m['hc2']['mean']*100:+.4f}pp、Vid{m['vidstg']['mean']*100:+.4f}pp，CI均跨0但不等价独立或弱交互证明，C−E与B−A时间收益仍双向正。实际root342064坐标/2048二次hash/189981独立scalar，完整GT support权重分解/真实未观测帧/旧32-新96-总128/两order/65严重负parent-contrast记录保留。{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree pass。人类取消写作，未写论文或新算法，不调P7_B/奖励/温度/名单/CURRENT。当前关闭只CPU机制实验，Query/LN/Slow Writeback的未来冻结及正式新方法全量未执行，不冒称整个后续请求完成；原队列最新优先级按人类实际回复/另存handoff，不伪造baseline global。见[实际P8root](</home/wwww/visual grounding/artifacts/stvg_opd_support_factorization_v1/ROOT_READBACK.json>)、[逐远端核验](</home/wwww/visual grounding/artifacts/stvg_opd_support_factorization_v1/FINAL_GITHUB_RECEIPT.json>)。\n\n"
    p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
    for a in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(a)==1;text=text.replace(a,a+note,1)
    p.write_text(text);archive(BASE/'archive_CPU_factorial_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings={n:sha(BASE/n) for n in ['ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json','GLOBAL_READOUT_BARRIER.json']}
    scope='complete finite P8 sealed-cache CPU2x2 prediction-support versus adaptation-support factorization; component deletion/freeze and full official main experiments remain unexecuted dependencies'
    write(BASE/'CPU_FACTORIAL_ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,cells=512,parents=256,logical_readouts=2048,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=True,time=time.time()))
    write(BASE/'CPU_FACTORIAL_FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'CPU_FACTORIAL_ROOT_CLOSING.json'),actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,whole_followup_request_complete=False,component_freeze_complete=False,full_official_main_complete=False,production_promoted=False,CPU_only=True,GT_read=True,time=time.time()))
    print('P8_FINITE_CPU_EXPERIMENT_ACTUAL_CLOSED',public['commit'])
if __name__=='__main__':run()
