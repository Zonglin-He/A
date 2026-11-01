"""Finalize an already independently audited/scored diagnostic, no new scoring."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round42';E=H/'experiment';I=H/'independent_audit'
def read(p):return json.loads(Path(p).read_text())
def main():
 s=read(I/'TASK_SUMMARY.json');a=read(I/'AUDIT_RESULT.json');receipt=read(I/'SCORING_RECEIPT.json');assert a['state']=='PASS_R42_INDEPENDENT_CPU_CONTRACT' and receipt['task_values']==1664
 names=['source','R29_direct','legacy_G','3B_point_compatible_fullmap','7B_point_compatible_fullmap','3B_point_compatible_oldsubset','7B_point_compatible_oldsubset','3B_som_compatible_fullmap'];table='| 方法 | Drone mIoU% | Quad mIoU% | 等权平均% |\n|---|---:|---:|---:|\n'
 for n in names:table+=f"| {n} | {s['drone'][n]['mIoU_percent']:.6f} | {s['quad'][n]['mIoU_percent']:.6f} | {s['balanced_mIoU_percent'][n]:.6f} |\n"
 aux='| 接口 | Drone Acc25/50% | Quad Acc25/50% |\n|---|---:|---:|\n'
 for n in ['source','3B_point_compatible_fullmap','7B_point_compatible_fullmap']:
  d=s['drone'][n];q=s['quad'][n];aux+=f"| {n} | {d['Acc25_percent']:.6f} / {d['Acc50_percent']:.6f} | {q['Acc25_percent']:.6f} / {q['Acc50_percent']:.6f} |\n"
 middle=(E/'REPORT_DRAFT.md').read_text().split('## 固定协议',1)[1].split('## 待独立结果补齐',1)[0]
 report='''# R42：固定 native-point 接口的 3B→7B 容量诊断

## 裁决与总体结果

**不采用本轮 7B point 配置，不接 TTA。** 主指标 mIoU 的两平台平均为 22.645214%，同接口 3B 为 22.740349%，差 −0.095135pp；Quad +0.650931pp，Drone −0.841201pp。旧 G 为 24.727373%，7B 低 2.082159pp。这个结论只针对固定提示、原图与 point/映射管线，不表示 7B 一般无效，也没有比较 7B 的其他输出接口。

'''+table+'''
其他指标并非全部下降，必须同时保留：Quad 的 Acc25/50 和 Drone 的 Acc50 提高；按预锁 mIoU 主指标不能据此切换采纳标准，也不能将本轮称为所有指标均退化。

'''+aux+'''
7B 相对 source 有 19 条 IoU 改善、3 条恶化；相对同接口 3B 有 24 条改善、31 条恶化，其余相同。全图中只有 25 条 source query 改变，3B 为 89 条。source 回退是固定方法的一部分，所有回退均计入总体；没有只报告接受子集。1664 个 IoU 与 1045 条有序负比较全部保留，重复对照及负比较不是独立数据。

## 弃权、映射与身份边界

7B 的 208 条原回答中，100 条显式 null、9 条解析失败；99 条给出结构合法点，其中 41 条越过既定图像像素边界。最终 fullmap 49 条映射（Drone20、Quad29），另有无 mask1、重叠3、空 3D 组5。完整映射覆盖为 49/208，原兼容 3B 为 160/208（Drone79、Quad81）。mapped 不等于主体身份正确，null 也没有改成错误的正伪标签。

同一预封 24 条身份记录中，18 明确、6 歧义。7B 的 18 明确中，5 条与原注释一致、1 条矛盾、12 条没有唯一 mask；原 3B 为 8/8/2。不能只取已回答的 5/6 与 3B 的 8/16 相比，然后宣布总体监督更可靠：回答覆盖明显不同，而且这些是非专家、存在历史暴露的暂定身份假设，不是新 gold。6 条歧义不判对错，身份从未被最终 IoU 改写。

7B 的 5 条一致身份在 fullmap 下均可映射，旧 subset 只有 2 条可映射；完整映射也让 1 条错误身份映射成功。它解决覆盖，不保证语义。原注释 SHA 保持 6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097，没有重新看图重标或按小子集救分。

## 固定协议'''+middle+'''
## 独立审计与封存

独立 CPU 审计 PASS：216 条 token/raw/input-binding 对照、216 次 processor 输入重构、416 个 full/old 映射和全部 1664 个原框精确一致；六个历史控制的任务值独立复现。解析器不执行原回答文本，拒绝重复键、非有限坐标、多对象歧义及超出既定 schema 的指令内容。输入 CPU 重构需要读取图像像素，但没有新增人工语义身份审查或生成模型。

task scoring 只有在审计 PASS、同一 prediction seal 的 root POSTSEAL 授权后执行。独立审计 16.05 秒，任务评分约 0.35 秒；这两个成本不在 124.58 秒模型阶段中。权重审计复用固定下载清单与严格 loader 绑定，不冒称再次独立散列 16.6GB。

有效 generate 显式 do_sample=False、num_beams=1、max_new_tokens=96、repetition_penalty=1.05。保存的基础 generation_config 中 do_sample=True/temperature 字段不代表本轮实际采样，不能混淆有效调用覆盖参数。没有 best-of、技术重答或答案择优。

全部数据是已暴露 A208，Drone2/Quad3 个既有场景，非新确认。场景表保留，不把208条当208独立场景，也不作一般显著性或容量普遍性声明。新增可更新参数为零、没有新3D框、没有源训练、没有TTA；本轮停止当前 point 配置，不改 parser、坐标单位、提示或门槛补救，不读取 U/T/FINAL。

报告与公开数据将保留所有 null、负例、18/6 身份边界、原下载 STOP 与后续资源修订。只公开自有代码、协议、匿名任务标量/状态与模型文件校验元信息；不公开权重、tokenizer 文件、第三方源码快照、token IDs、原回答、原 caption、坐标、mask、点云、真实场景路径或媒体。

证据入口：`../independent_audit/{AUDIT_RESULT,SCORING_RECEIPT,TASK_SUMMARY,COVERAGE_UNKNOWN_SUMMARY,IDENTITY24_SUMMARY}.json`、全部逐例结果/负比较与匿名场景表；原运行、下载和预算证据见相应封存 SHA。
'''
 (E/'REPORT.md').write_text(report)
 decision={'state':'NO_ADOPT_FIXED_7B_NATIVE_POINT_CONFIGURATION_NO_TTA','run':'complete216calls208observations8arms','measurement':'independentCPUcontractand1664IoU_PASS','primary':'prespecifiedbalancedmIoU','balanced_mIoU_percent':s['balanced_mIoU_percent'],'delta7B_vs_matched3B_pp':s['balanced_mIoU_percent']['7B_point_compatible_fullmap']-s['balanced_mIoU_percent']['3B_point_compatible_fullmap'],'other_Acc25_50_improvements_retained_not_switched_primary':True,'7B_fullmap_mapped':49,'3B_fullmap_mapped':160,'null_preserved':100,'all1664values_1045negativecomparisons_retained':True,'provisional_identity_clear18_ambiguous6':'5consistent1contradicts12unknown;notgold/coverageaware','no_general_7B_ineffectiveness_claim':True,'samefixedpoint_interface_only':True,'no_TTA_newboxes_source_training_FINAL_U_T':True,'no_new_method_or_deployment_adoption':True,'GPU_CPU_released':True}
 (E/'DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');print('R42_REPORT_DECISION_COMPLETE')
if __name__=='__main__':main()
