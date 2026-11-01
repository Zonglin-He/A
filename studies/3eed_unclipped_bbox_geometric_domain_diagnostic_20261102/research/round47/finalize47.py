"""Write a bounded author report from fixed independently audited scalar files."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'research/round47/experiment';H=R/'research/round47/independent_audit'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=read(H/'AUDIT_RESULT.json');v=read(H/'SCORING_RECEIPT.json');s=read(H/'TASK_SUMMARY.json');c=read(H/'DOMAIN_CONTRASTS.json');assert a['state']=='PASS_R47_INDEPENDENT_CACHED_DOMAIN_CONTRACT' and v['task_values']==6032 and v['decision']=='STOP_FIXED_DOMAIN_CONFIGURATION';m=s['balanced_mIoU_percent']
 decision={'state':'STOP_FIXED_DOMAIN_CONFIGURATION_NO_MORE_COORDINATE_OR_QUALITY_GRID','primary':'7B_bbox_domain_rectIoU_fullmap','primary_balanced_mIoU_percent':m['7B_bbox_domain_rectIoU_fullmap'],'primary_minus_strict_pp':c['balanced']['7B_rectIoU_domain_minus_strict_pp'],'primary_minus_legacy_pp':m['7B_bbox_domain_rectIoU_fullmap']-m['legacy_G'],'secondary_center_gain_vs_strict_pp':c['balanced']['7B_center_domain_minus_strict_pp'],'secondary_not_reselected_as_primary':True,'no_auto_TTA_or_7B_FINAL_admission':True,'all_exposed_A208_development':True,'original_R45_strict_contract_and_results_unchanged':True,'scientific_limit':'fixed cacheddomainreadout negative,not all7B/nativegrounding falsification;geometryadmissibility doesnotcertify promptunitcompliance oridentity','prediction_seal_sha256':v['prediction_seal_sha256'],'audit_sha256':sha(H/'AUDIT_RESULT.json'),'score_sha256':sha(H/'SCORING_RECEIPT.json'),'no_new_model_or_optimizer_calls':True,'GT_used_only_by_independent_postseal_scoring_not_feedback':True,'no_FINAL_U_T_access':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (E/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
 names=[('Frozen source','source'),('旧复杂 G','legacy_G'),('3B point（缓存）','point_compatible_fullmap'),('7B point（缓存）','7B_point_compatible_fullmap'),('3B bbox center strict/domain','bbox_fullmap'),('3B bbox rectIoU strict/domain','bbox_IoU_fullmap'),('7B bbox center strict','7B_bbox_center_fullmap'),('7B bbox center domain（次要）','7B_bbox_domain_center_fullmap'),('7B bbox rectIoU strict','7B_bbox_rectIoU_fullmap'),('**7B bbox rectIoU domain（预设主项）**','7B_bbox_domain_rectIoU_fullmap')]
 table='| 方法 | Drone mIoU % | Quad mIoU % | 平均 % |\n|---|---:|---:|---:|\n'+''.join(f"| {label} | {s['drone'][k]['mIoU_percent']:.6f} | {s['quad'][k]['mIoU_percent']:.6f} | {m[k]:.6f} |\n" for label,k in names)
 extra='| 7B 读出 | Drone Acc@25/50 % | Quad Acc@25/50 % | 相对 source 改善/恶化条数 |\n|---|---:|---:|---:|\n'
 for mode in ['center','rectIoU']:
  k=f'7B_bbox_domain_{mode}_fullmap';d=s['drone'][k];q=s['quad'][k];extra+=f"| domain {mode} | {d['Acc25_percent']:.6f} / {d['Acc50_percent']:.6f} | {q['Acc25_percent']:.6f} / {q['Acc50_percent']:.6f} | {d['improved_vs_source']+q['improved_vs_source']} / {d['harmed_vs_source']+q['harmed_vs_source']} |\n"
 report=f'''# R47：原生 bbox 几何域敏感性诊断

**停止当前固定域扩展。预先指定的 7B unclipped rectangle-IoU 主项为 {m['7B_bbox_domain_rectIoU_fullmap']:.6f}%，比原 strict 读出低 {-c['balanced']['7B_rectIoU_domain_minus_strict_pp']:.6f} 个百分点，低于旧 G 的 {m['legacy_G']:.6f}%。不再扩展裁剪、单位、坐标阈值、quality 或模型网格，也不自动接 TTA。**

次要 center 读出比 strict 提高 {c['balanced']['7B_center_domain_minus_strict_pp']:.6f} 个百分点，达到 {m['7B_bbox_domain_center_fullmap']:.6f}%；这项收益保留，但仍低于旧 G，不反选为本轮 primary。结论限定于该固定缓存接口；不是“所有7B或原生grounding无效”。

## 唯一变化

沿用 R39 的3B、R45的7B各208条原始 bbox 回答、确切 grid、完整描述、独立 mask 库及原3D候选组。只允许原解析已通过有限正 xyxy 面积、而唯一失败原因是四角越出图像范围的输出进入几何读出。3B恢复2条，7B恢复115条；原JSON/schema/坐标数值、单位和面积不变。原115条仍属于原 prompt 声明域不合规，不能因几何可计算而改称输出正确。

中心仍须在图内、唯一命中mask；矩形使用完整原面积的标准 unclipped IoU，在全部mask的detector矩形中选唯一正最大值。最大mask无3D组时保留UNKNOWN，不跳次优；共59个此类最大空组事件保留。没有裁剪、重排四角、0–1000切换、最近候选补救、新阈值、模型重答或参数更新。

## 全体结果

两平台各104条，共208条已暴露A开发观测。29组中21组原strict缓存控制、8组新域读出；所有6032个任务IoU和16135个有序负比较完整保留，后者是重复方法比较，不是独立样本。

{table}
{extra}
3B四个新域读出的任务值均与各自strict完全一致。7B主项相对strict在Drone为 +0.766725pp，在Quad为 −1.125022pp，平均 −0.179149pp；center次项分别 +2.077828/−0.082348pp。扩大可映射覆盖没有稳定转化为更好目标选择。

域读出的7B fullmap覆盖为 center94/208、rectIoU137/208；原strict分别46/53。覆盖只是可进行映射，不是身份正确率。主项相对source有33条改善、26条恶化；center有26条改善、15条恶化，全部正负例均保存。不可只看接受子集或单例救回解释总体。

## 独立审计、失败历史与成本

- 原416条3B/7B回答未改变；21个strict候选索引/框、4368个原任务IoU全部一致。
- 独立重新计算1664个新读出，全部mask IoU向量最大误差0；1032个原strict有效读出保持一致，坏JSON、非有限和非正面积未恢复。
- 首次独立audit在诊断字符串接口失败：旧独立parser对 parse_failure 省略JSONDecodeError异常名称，原status/value一致。保留原代码、锁与失败日志；仅取消对该缺失诊断名称的比较，outside/invalid的决策原因仍逐项比较。没有修改预测、schema、几何或方法门；第二次audit通过后才进行唯一一次任务评分。
- 新模型调用、GPU工作、生成、优化步骤均为0；仅缓存CPU预测与独立封后GT评分。独立audit约1.83秒、评分约0.83秒；这些不是完整历史前端/VLM成本。不存在“无GT评估”，GT仅在封存后的独立任务评分使用，不进入反馈或方法选择。

## 边界与决定

本轮评估仍固定先前GT-free名词读出source与相同候选，不偷偷改用metadata目标类别span。几何域有效与语义身份、坐标单位、原提示词遵从是不同条件。越界恢复并未证明原生框一定描述完整物体；错误实例和单位误差仍可能存在。

A208已暴露、场景相关，场景结果只作描述，不计算跨场景显著性或把重复行当独立证据。未访问 FINAL/U/T；不将本轮7B接口纳入最终确认。原R45 strict合同、结果和primary完整保留。

**决定：关闭当前固定域扩展；保留center次项及所有负例作为诊断证据，不继续坐标/quality网格，不据其启动TTA。**
'''
 (E/'REPORT.md').write_text(report);print('R47_AUTHOR_REPORT_DECISION_WRITTEN')
if __name__=='__main__':main()
