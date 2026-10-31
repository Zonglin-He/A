"""Audit supplemental artifacts and append bounded conclusions, no model calls."""
from pathlib import Path
import sys,collections,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round18'))
from common import EXP,read,dump,sha,verify
def main():
 paths=['supplement32_attempt001','supplement_controls32_attempt003'];ss={p:verify(ROOT/'crossplatform/round18'/p) for p in paths}
 for p in ('scoredC32_attempt001','scoredCcontrols32_attempt001'):verify(EXP/p)
 c=read(EXP.parent/'input/C32_PRIVATE.json');other=[]
 for p in (ROOT/'research/round15/input/SOURCE96_GPU_INPUTS.json',ROOT/'research/round15/input/REGRESSION140_GPU_INPUTS.json',EXP.parent/'input/ALL96_PRIVATE.json'):other+=read(p)
 assert len(c)==32 and collections.Counter(r['platform'] for r in c)=={'drone':16,'quad':16};assert not {r['id'] for r in c}&{r['id'] for r in other}
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv}
 def content(r):
  s=lookup[sha(r['imagepath'])];assert s['point_file_sha256']==sha(r['pointpath']);return (s['decoded_pixel_sha256'],s['canonical_XYZ_sha256'])
 cc=[content(r) for r in c];oo=[content(r) for r in other if r['platform'] in ('drone','quad')]
 for k in (0,1):assert len({x[k] for x in cc})==32 and not {x[k] for x in cc}&{x[k] for x in oo}
 trace=read(ROOT/'crossplatform/round18/supplement32_attempt001/TRACE_PRIVATE.json');stats={}
 for name in trace[0]['diagnostics']:
  dd=[r['diagnostics'][name] for r in trace]
  for d in dd:
   assert d['nonselected_parameters_unchanged'];assert (d['parameter_delta_L2']>0)==(d['steps']>0)
   if name.startswith('MEMO') and d['steps']:assert d['common_groups']>=2 and min(d['coverage'])>=.5
  stats[name]={'n':len(dd),'optimizer_steps':sum(d['steps'] for d in dd),'nonzero_updates':sum(d['parameter_delta_L2']>0 for d in dd),'selected_parameters':sorted({d['selected_parameters'] for d in dd})}
 sc=read(EXP/'scoredC32_attempt001/SUMMARY.json')['C32_postprimary_official_config_diagnostic'];ctrl=read(EXP/'scoredCcontrols32_attempt001/SUMMARY.json')['C32_postscore_omitted_control_capture'];merged={p:dict(sc[p],**{k:v for k,v in ctrl[p].items() if k!='source'}) for p in sc}
 for p in sc:assert sc[p]['source']['mIoU_percent']==ctrl[p]['source']['mIoU_percent']
 for scored in ('scoredC32_attempt001','scoredCcontrols32_attempt001'):
  rows=read(EXP/scored/'ROWS_PRIVATE.json');summ=read(EXP/scored/'SUMMARY.json');assert len(rows)==32
  for sp,platforms in summ.items():
   for p,arms in platforms.items():
    rr=[r for r in rows if r['platform']==p]
    for a,s in arms.items():assert s['n']==len(rr) and abs(s['mIoU_percent']-100*np.mean([r['ious'][a] for r in rr]))<1e-12
 replay=ss['supplement_controls32_attempt003'];assert replay['source_bitexact'] and replay['max_adapted_replay_error_m']<=1e-4
 audit={'state':'R18_SUPPLEMENT_CPU_REPLAY_PASSED','C32_unique_current_ids_and_decoded_RGB_XYZ':True,'no_overlap_with_R15_source96_regression140_R17_AB96':True,'all_target_VAL_historical_exposure':True,'all32_negative_rows_kept':True,'all_prediction_and_score_seals_verified':True,'scope_and_actual_updates_verified':True,'step_counts':stats,'source_predictions_repeated_bitexact':True,'adapted_replay_bitexact':replay['repeat_bitexact'],'max_adapted_replay_error_m':replay['max_adapted_replay_error_m'],'declared_bound_m':1e-4,'failed_bitexact_attempts_retained':2,'original_C_predictions_not_overwritten':True,'controls_captured_after_C_task_scores_exposed':True,'control_capture_not_independent_confirmation':True,'artifact_seal_hashes':{p:sha(ROOT/'crossplatform/round18'/p/'SEALED.json') for p in paths}}
 dump(EXP/'SUPPLEMENT_CPU_AUDIT.json',audit);dump(EXP/'C32_COMPARISON.json',{'status':'Supplementary descriptive comparison,not primary replacement','platforms':merged,'steps':stats,'control_capture_timing':'After C scoring; fixed operations; bitexact source and bounded FP32 adapted replay; no configuration selection'})
 report=EXP/'REPORT.md';s=report.read_text();assert '## C32官方配置补充' not in s
 lines=['','## C32官方配置补充（独立列示，不替换主表）','','主表评分后，核对MEMO官方`imagenet-exps/script_test_in.py`发现Transformer/RVT分支使用AdamW 1e-5、weight_decay=.01，主表采用其ResNet SGD 2.5e-4配置。因此固定比较两套原始优化配置，并另测只更新LN/GN、保持source BN的SAR；不是按C成绩挑选超参数。C32按输入哈希选择，每平台16条，ID与实际解码RGB/XYZ均排除R15 source96/regression140和R17/R18 A/B96；仍在历史暴露目标VAL、原有1/2场景内，不是盲测。','','| 方法 | Drone mIoU / Acc25 / Acc50 % | Quad mIoU / Acc25 / Acc50 % |','|---|---:|---:|']
 for a in ('source','SAR_LN_only','MEMO_C_BN_prior_only','MEMO_C_augmentation_only','MEMO_SGD_25e5','MEMO_AdamW_1e5'):
  cells=[merged[p][a] for p in ('drone','quad')];lines.append('| '+a+' | '+' | '.join(f'{v["mIoU_percent"]:.6f} / {v["Acc25_percent"]:.2f} / {v["Acc50_percent"]:.2f}' for v in cells)+' |')
 lines+=['','内部名`MEMO_SGD_25e5`实际学习率为2.5e-4，`MEMO_AdamW_1e5`实际为1e-5，不应按字符串误读。','','| C32 Quad方法 | 样本均值增益pp | 两场景等权增益pp | 改善/恶化/不变 |','|---|---:|---:|---|']
 for a in ('MEMO_C_augmentation_only','MEMO_SGD_25e5','MEMO_AdamW_1e5'):
  v=merged['quad'][a];lines.append(f'| {a} | {v["gain_pp"]:+.6f} | {v["equal_scene_gain_pp"]:+.6f} | {v["gains"]}/{v["harms"]}/{v["ties"]} |')
 lines+=['','C32的SAR LN-only消除了本轮BN统计切换的大退化，任务改善仍近零。MEMO两配置在Quad样本均值为正，但两场景等权增益均为负，Acc50由6.25%降至0，不能宣称稳定跨场景提升。Drone两配置均退化；不以最佳C行替换原A/B结果。',f'',f'C32原runner漏存已计算的增强重排与BN先验输出。补录发生在C任务评分暴露之后，仅重复固定计算，不变更输入/超参/接收门。两次逐位复现断言失败已封存保留；全部32条source重复逐位一致，更新后框最大数值差为{replay["max_adapted_replay_error_m"]:.9g}米，低于事先记录的1e-4米工程容差。其精确底层原因未建立，不能称反向过程逐位可复现。原始C预测从未覆盖。补录控制只作描述性对照，不作为独立确认。','', '**最终裁决：外部baseline接口真实运行完成，当前不采用新TTA方法；Reg仅为启发移植，完整原方法复现仍缺核验接口。所有128条当前观测及其负例、在线补充与同信息对照保留。没有继续排队的GPU任务。**']
 report.write_text(s+'\n'.join(lines)+'\n')
 d=read(EXP/'DECISION.json');d.update(state='R18_PRIMARY96_SUPPLEMENT32_COMPLETED_SCORED_AUDITED',supplement_C32_completed=True,supplemental_same_information_controls_completed=True,controls_capture_after_C_scores=True,C32_official_AdamW_not_primary_replacement=True,supplement_audit_sha256=sha(EXP/'SUPPLEMENT_CPU_AUDIT.json'));dump(EXP/'DECISION.json',d)
 dump(EXP/'RUN_STATE.json',{'state':d['state'],'adopted':False,'current_scoring_complete':True,'no_new_GPU_queued':True,'decision_sha256':sha(EXP/'DECISION.json'),'primary96':True,'supplement32':True})
 state=read(ROOT/'research/STATE.json');state['round18_snapshot'].update(supplement_C32=True,supplement_control_capture_completed=True,new_GPU_queued=False);dump(ROOT/'research/STATE.json',state)
 with (ROOT/'research/ARCHIVE.md').open('a') as f:f.write('\nR18 supplement: newC32 officialMEMO optimizer and SAR LN-only diagnostic completed, same-information omitted controls recovered with all source predictions bitexact and bounded FP32 adapted replay. Quad sample-mean MEMO gains do not survive equal-scene weighting; no adoption. All negative rows retained.\n')
 print('SUPPLEMENT_FINALIZED',stats,'replay',replay['max_adapted_replay_error_m'])
if __name__=='__main__':main()
