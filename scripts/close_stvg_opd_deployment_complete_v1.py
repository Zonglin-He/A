"""Actual finite deployment timing closure; remaining LN and audited cost stay open."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_stvg_opd_deployment_latency_v1 import BASE,verify
from scripts.close_stvg_opd_p7_phase_v1 import archive
def run():
    verify();v=read(BASE/'ROOT_VISUAL_REVIEW.json');p=read(BASE/'FINAL_GITHUB_RECEIPT.json');r=read(BASE/'ROOT_COMPLETE_READBACK.json')
    assert v['status']==p['status']==r['status']=='pass' and v['actual_pixel_view'] and p['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    assert r['actual_timed_query_paths']==128 and read(BASE/'PUBLIC_SCALAR_AUDIT.json')['status']=='pass'
    for rel,b in v['PDF_files'].items():assert sha(ROOT/rel)==b['sha256'] and (ROOT/rel).stat().st_size==b['bytes']
    note=f"**2026-10-13｜原128 Full/Frozen配对部署计时实际有限FINAL。** 原每域16固定query/两repeat/64完整GPU资格fit真实root4852728轨迹＋134208trace-off状态坐标pass，随后128真实timed paths全部383104原预测/状态/writeback坐标和896stage sum独立root、真实一PDF一页pixel view/完整匿名timings及所有无信息路径/public/archive实际通过。HC Frozen/Full两repeat逐query均值的median=2.578/4.417秒、paired增加median1.851秒、ratio1.687；Vid=2.415/3.478秒、paired增加1.105秒、ratio1.449；不得用两median相减替配对增加。资格实际DINO112调用，timed Full实际224（HC128/Vid96），Vid8重复路径无观察/DINO/反向，HC4与Vid8 Full重复路径0backward均保留。GT_read=false，全部原源/帧/精度/Full/专家/fit/优化器/读出不改，仅原O012存储与Full inactive expert暂存CPU的开销实计timer，原S暂停GPUcontext仍常驻；是说明缓存/硬件/本process显存条件的本地测量，不是去掉专家成本或最终生产throughput。{p['file_count']}文件/{p['bytes']}bytes在main/research共同{p['commit']}、tree{p['tree_sha']}逐远端实际pass。尚不关闭LN768/32 audited cost及后续决定，不恢复剩余baseline，不新gate/因果方法/论文/CURRENT晋升。\n\n"
    path=ROOT/'docs/RESEARCH_HISTORY.md';text=path.read_text()
    for a in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(a)==1;text=text.replace(a,a+note,1)
    path.write_text(text);archive(BASE/'archive_deployment_actual_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings={n:sha(BASE/n) for n in ['ROOT_QUALIFICATION_READBACK.json','ROOT_COMPLETE_READBACK.json','ROOT_PUBLIC_TIMING_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']}
    scope='complete original32 matched queries/128 actual paired deployment paths/64 complete qualification fits/full output-state-root/actual PDF pixel view/all timing outcomes/public/archive; not LN768 or32 audited cost'
    write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,actual_timed_query_paths=128,unique_queries=32,
        actual_qualification_GPU_fits=64,actual_timed_DINO_calls=224,PDFs=1,pages=1,actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,
        CPU_only=True,GT_read=False,time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,LN768_complete=False,cost32_complete=False,
        whole_followup_request_complete=False,production_promoted=False,CPU_only=True,GT_read=False,time=time.time()))
    print('DEPLOYMENT_ACTUAL_FINITE_SCOPE_CLOSED',p['commit'])
if __name__=='__main__':run()
