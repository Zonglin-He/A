"""Close an actual P7 phase after independent root, pixel view and remote verification."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *

def archive(folder,receipt_name):
    folder.mkdir(parents=True,exist_ok=False);steps=[]
    for job in ['check','snapshot','check']:
        name=f'{len(steps)}_{job}';p=folder/(name+'.json')
        z=subprocess.run([str(PYTHON),'-B','scripts/research_archive.py',job],cwd=ROOT,capture_output=True,text=True)
        p.write_text(z.stdout);(folder/(name+'.stderr.txt')).write_text(z.stderr)
        assert z.returncode==0,z.stderr+z.stdout[-1500:]
        d=json.loads(z.stdout);assert d['status']==('snapshot_written' if job=='snapshot' else 'pass')
        steps.append(dict(job=job,status=d['status'],path=str(p.relative_to(ROOT)),sha256=sha(p)))
    write(folder.parent/receipt_name,dict(status='pass',scope='actual research archive check-snapshot-check',steps=steps,CPU_only=True,GT_read=False,time=time.time()))

def run(phase):
    verify();ns=A if phase=='A' else B
    root=read(ns/'ROOT_READBACK.json');view=read(ns/'ROOT_VISUAL_REVIEW.json');public=read(ns/'FINAL_GITHUB_RECEIPT.json')
    assert root['status']==view['status']==public['status']=='pass' and view['actual_pixel_view']
    assert public['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    for rel,binding in view['PDF_files'].items():assert sha(ROOT/rel)==binding['sha256'] and (ROOT/rel).stat().st_size==binding['bytes']
    if phase=='A':
        decision=read(A/'P7_B_BRANCH_DECISION.json');scope='P7_A original128 historical parents each/twoorders/all512 clean cells/all4096 frozen temporal-spatial readouts/no new spatial fits'
        note=f"**2026-10-12｜P7-A有限扩大匹配阶段实际关闭（不代表P7-B或S/C/N全部任务完成）。** 原512clean/4096八读出全部GPU冻结capture、GT后dense/独立完整来源与统计、两PDF四页实际pixel view通过；全部正负匿名结果及完整公开源码{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际通过。新增96两源时间OPD收益均正且CI不跨零，原分支P7_B_eligible={decision['P7_B_eligible']}；交互仍跨零、Vid cap优于OPD保留。下一步仅同原名单/Full-before/超参/K4一次条件B，尚无B实际结果；S原同进程保存10067prefix恢复，C/N依赖与半小时Luna ACTIVE保持。"
    else:
        decision=read(B/'CANDIDATE_MECHANISM_DECISION.json');scope='P7_B fixed512 same-original-Full-before observation-allocation experiment/all1024 complete C-D fits/A-B exact reuse/all2048 dense readouts/all negative and support diagnostics'
        note=f"**2026-10-12｜P7-B单次时间条件化专家观察实验实际关闭。** 固定512clean/1024新C-D完整fit及16完整资格repeat、真实专家pack预算、全global后GT/完整math-state-source-input/dense/独立bootstrap、全部实际PDF pixel view与匿名完整正负公开通过；{public['file_count']}文件/{public['bytes']}bytes在main/research共同{public['commit']}、tree{public['tree_sha']}逐远端bytes/SHA256/Gitblob/mode/tree实际通过。新增96两源coupling及真实未观测帧共同支持判定={decision['misallocation_hypothesis_supported_in_both_added96']}；诚实保留Direct比较和严重负尾，未双向joint/时间参数TTA/生产晋升/全官方query扩大。S同原进程optimizer/prefix实际恢复；旧S/C/N完整任务未齐，半小时Luna继续ACTIVE。"
    note+=' 见[实际root](</home/wwww/visual grounding/artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_'+phase+'/ROOT_READBACK.json>)、[真实图view](</home/wwww/visual grounding/artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_'+phase+'/ROOT_VISUAL_REVIEW.json>)、[逐远端公开](</home/wwww/visual grounding/artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_'+phase+'/FINAL_GITHUB_RECEIPT.json>)。\n\n'
    p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text()
    for anchor in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert s.count(anchor)==1;s=s.replace(anchor,anchor+note,1)
    p.write_text(s)
    archive(ns/'archive_phase_closing','ARCHIVE_VERIFICATION_RECEIPT.json')
    bindings={name:sha(ns/name) for name in ['ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json','PRIORITY_RESUMPTION_RECEIPT.json']}
    write(ns/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,phase='P7_'+phase,complete_current_phase=True,whole_S_C_N_task_complete=False,CPU_only=True,GT_read=True,time=time.time()))
    write(ns/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,phase='P7_'+phase,root_closing_sha256=sha(ns/'ROOT_CLOSING.json'),actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,whole_S_C_N_task_complete=False,production_promoted=False,full_official_query_expansion=False,time=time.time()))
    print('P7_PHASE_ACTUAL_CLOSED',phase,public['commit'])

if __name__=='__main__':run(sys.argv[1])
