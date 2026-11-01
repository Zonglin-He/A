"""Report sealed R39 center versus R40 standard rectangle association."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round40';EXP=HERE/'experiment';R39=ROOT/'research/round39'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 a39=read(R39/'independent_audit/AUDIT_RESULT.json');a40=read(HERE/'independent_audit/AUDIT_RESULT.json');assert a39['state']=='PASS_R39_ORIGINAL_CENTER_PIPELINE' and a40['state']=='PASS_R40_CACHED_RECTANGLE_ASSIGNMENT'
 s=read(EXP/'TASK_SUMMARY.json');c=read(EXP/'COVERAGE_UNKNOWN.json');names={'source':'Frozen source','R29_direct':'R29 direct','legacy_G':'旧G','point_strict_fullmap':'原strict point＋fullmap','point_strict_oldsubset':'原strict point＋旧subset','som_strict_fullmap':'原strict SoM＋fullmap','som_strict_oldsubset':'原strict SoM＋旧subset','point_compatible_fullmap':'兼容point＋fullmap','point_compatible_oldsubset':'兼容point＋旧subset','som_compatible_fullmap':'兼容SoM＋fullmap','som_compatible_oldsubset':'兼容SoM＋旧subset','bbox_fullmap':'R39 bbox中心＋fullmap','bbox_oldsubset':'R39 bbox中心＋旧subset','bbox_IoU_fullmap':'R40 bbox矩形IoU＋fullmap','bbox_IoU_oldsubset':'R40 bbox矩形IoU＋旧subset'}
 table='\n'.join('| '+names[arm]+' | '+f"{s['drone'][arm]['mIoU_percent']:.6f} | {s['quad'][arm]['mIoU_percent']:.6f} | {v:.6f} |" for arm,v in s['balanced_mIoU_percent'].items())
 coverage='\n'.join('| '+names[arm]+' | '+f"{c['drone'][arm]['mapped']}/104 | {c['quad'][arm]['mapped']}/104 | {c['drone'][arm]['mapped']+c['quad'][arm]['mapped']}/208 |" for arm in ['point_compatible_fullmap','som_compatible_fullmap','bbox_fullmap','bbox_oldsubset','bbox_IoU_fullmap','bbox_IoU_oldsubset'])
 isummary=read(EXP/'IDENTITY_PUBLIC_SUMMARY.json');assert isummary['R39_join_sealed'] and isummary['annotations_unchanged'];itext=isummary['report_text']
 report=f'''# R39/R40：原生二维框输出与mask关联诊断

**状态：R39原bbox中心版本、R40缓存矩形IoU关联版本均完成封存、独立CPU审计和封存后评分。** 两轮只产生G的直接选框结果，没有学生参数适应或新三维框。固定A208仍是已暴露开发集，不是新增独立确认；原R37/R38所有封存结果及公开导出保持不变。

**总体结论：bbox中心20.733712%，标准矩形IoU关联21.038120%，较中心＋0.304408pp；二者仍低于兼容point22.740349%和旧G24.727373%。** 现有二维框范围能修复部分中心关联，但没有证明它是总体误差的主要原因。停止这两个固定读出配置的TTA接入，不以身份小子集或少数恢复案例反选采纳。

## 1. 固定观测与两个唯一变化

三维候选继续来自同一Waymo-only checkpoint，目标为Drone／Quad；这里是跨平台开发诊断，不是源平台训练重新开展或未暴露测试。R39沿用同一冻结Qwen2.5-VL-3B、相同原图、完整描述、system边界、greedy/max96、processor像素预算及绝对grid×14单位，只将所请求的原生point换为单个`bbox_2d` xyxy。允许单对象或单对象列表、唯一已知bbox键和可选惰性字符串label；多对象／多位置、重复键、非有限数、bool、数值字符串、任意尾文本等拒绝。xyxy必须有严格正面积且处于声明输入范围，不裁剪、不换序、不选择1000单位。

R39只将二维框中心作为身份探针，经固定逐轴逆变换和floor像素检验，必须落入恰好一个原独立mask，再进入原full／旧subset三维组，固定原source最高分query。框中心不是三维中心，也不把二维框转成新三维框。原11条缓存对照的query和框完全不变。

R40不重新调用模型、重解析schema或改位置。它将同一合法bbox按原grid逆变换到原图，与**所有**原detector矩形计算标准rectangle IoU。严格唯一且正的最大值提名mask；全零、精确并列或无效／不完整矩形库为unknown。先在全部mask中取最大值，再查三维组；不得为找可映射组而跳过无组mask、改选次优。使用detector矩形而非真实实例边界，因此仍可能被遮挡、邻物和背景影响，不能称新算法或更强bbox模型。原13臂精确保留，只新增两个IoU关联臂。

两轮均复用原mask、source256三维框和分数、完整／旧subset组及固定代表，不以GT、颜色类别或当前q0筛mask；所有unknown保留source回退。新读出提供信息不等于认证身份，完整组也不保证含有好框。

## 2. 同一A208上的全部总体指标

mIoU%，Drone和Quad各104条，平均为平台等权。共享观测及重复缓存臂不算独立样本。

| 方法 | Drone | Quad | 平台等权平均 |
|---|---:|---:|---:|
{table}

中心bbox相对source＋1.070149pp，却低于point2.006637pp、旧G3.993661pp。矩形IoU相对source＋1.374557pp，仍低于point1.702229pp、旧G3.689253pp。两个bbox版本在Drone均低于source，不能以Quad正收益掩盖平台退化，也不能因平台平均为正就自动接入参数更新。

矩形IoU相对中心的改善只有Drone＋0.148653pp、Quad＋0.460163pp；逐例为 **4条改善、3条恶化、201条不变**。它确实使用了中心之外的bbox范围，但总体增量小。旧subset下，两种bbox读出任务指标完全相同，不能把它描述为普遍更强的关联。

| 方法 | Drone合法3D映射 | Quad合法3D映射 | 全208 |
|---|---:|---:|---:|
{coverage}

中心映射164/208，矩形IoU映射176/208。更高覆盖同时保留错误实例：最大IoU提名中26条没有full三维组（Drone19、Quad7），均回退，未跳次优。原4条坏JSON和2条越界继续unknown。全量数据没有触发全零或精确并列分支；独立CPU合成合同单独覆盖这些分支，不能据未触发宣称不会发生。

## 3. 机制子集只作解释，不能当整体选择门

子集在任务GT之前由原中心unknown与两方法提名冲突定义，封存后才评分：

| 子集 | Drone | Quad |
|---|---|---|
| 原中心unknown | 28条，2好1坏，均值＋0.490093pp | 16条，任务IoU均不变 |
| 中心和IoU提名不同mask且两者均有提名 | 6条，2好2坏，均值＋2.341825pp | 2条，1好0坏，均值＋23.928495pp |

这两个子集可能重叠，不能相加当额外样本或独立证据；Quad两条中的大幅恢复不能代替全A208的失败对照。它们支持“中心压缩有时丢失有用范围信息”，不支持“只要改关联就解决整体主体理解”。原R37坐标合成诊断差、mask污染、bbox自身定位、主语／关系理解与三维候选质量仍是竞争解释。

## 4. 同一预封身份核验

{itext}

身份注释不是训练标签或在线接受门，报告代理没有新看图、改注释或按IoU改身份。小子集好于point不能推翻全208对point／旧G的负对照。

## 5. 真实生成失败、修复和调用预算

R39最初attempt001第一次实际生成完成后，在可选图像散列序列化处因缺少`np`导入报NameError，raw尚未保存而丢失。该次真实模型调用必须计入，原failure、原ledger和原计划均保留；原局部`newcalls=0`只是函数未正常返回后的计数，不能据此说没有生成。丢失回答的tokens和时间无法恢复，明确为unknown，没有伪造或省略成本。

在本轮任务评分前，root单独授权纯技术修复和209硬预算：补导入，并在解析／图像散列前将唯一raw及模型元数据写临时文件、flush/fsync后原子replace；后续异常时优先复用该stage，不重新调用模型。固定prompt、模型、schema、单位、映射、候选和阈值不变；同一输入的技术重跑不是从多个可用答案择优。不能声称未保存的旧回答与新回答内容相同。

attempt002的208个raw stage全部可回放，stage与最终call中的原始回答及元数据逐项相同，写入早于下游解析／图像散列；没有残留临时文件。该证据验证本次产物和代码顺序，不是所有断电／文件系统崩溃情形的耐久性保证。

实际生成总数 **209＝1次旧raw丢失＋208次成功**。12条真实预检已包含在208中，正式阶段只新增196；11条有效／null、9条合法映射的技术预检不等于完美bbox能力门。成功回答202条parsed、4条坏JSON、2条越界，无额外模型合成控制或语义重试。

208成功调用输入254003 token、视觉208000、输出6155，生成时间约143.9958秒；另1次丢失调用的tokens／时间unknown。正式阶段回执151.5072秒只覆盖其指定阶段，不是全部端到端部署延迟。R40新增VLM、GPU和TTA调用均为0，只做缓存几何关联；旧R39生成成本不因缓存重放而消失。

## 6. 独立审计与历史边界

R39独立复算208回答→安全bbox解析→中心逆变换→唯一mask→固定代表，验证原11控制query／框逐位一致；13臂2704IoU、1733条负比较、全部指标／覆盖／成本／场景／contrast复现。模型11文件hash和R37完整几何审计按已验证链复用，不把复用说成新模型加载或新全场景重算；冻结证据沿用生成器eval、requires_grad=False、参数版本／buffer散列及末尾unchanged回执，CPU不独立读取GPU参数内存。

R40独立重算所有mask的矩形IoU向量，最大误差0，验证全部inventory和candidate rectangles逐项相同；26个最高mask无三维组时不改选次优。15臂3120IoU、2469条负比较和所有子集／场景／contrast复现。两轮seal均早于各自评分。安全正控覆盖多对象、代码／恶意位置、数值类型、重复键、越界不裁剪及几何并列／全零／无矩形库的unknown规则。

原R37/R38的loader零调用失败、旧蓝区1/2失败、新8控制1/8命中、数据回答前root测量门重审、文档hash快照修复和所有严格／兼容结果继续保留。bbox新实验不能擦掉原坐标能力诊断的不确定性，也不复现官方RefCOCO协议。固定官方cookbook示例引用7B，而本轮是本地3B及自定义3EED完整描述提示；不能把本轮解释为官方7B REC能力复现。官方plot helper的截断修复或literal_eval没有进入本轮安全解析。原生bbox与标准矩形匹配分别是接口比较和经典几何对照，不是方法新颖性依据。

全部2704＋3120任务标量、1733＋2469负方法／参考比较、unknown和坏例导出，但复录旧臂不能当新增独立证据。A208只有2个Drone、3个Quad历史场景且已反复开发暴露；固定场景重采样仅作描述，不作新独立确认或总体显著性。没有U/T/FINAL新读取、推理或评分。

## 7. 裁决

**运行及测量有效；bbox输出和范围关联均未超过point及旧G，不能替代当前精度参照，也不接入回归层或匹配层TTA。** 标准矩形IoU小幅改善中心关联，但仍留下误身份和无三维组问题。保留所有直接无更新对照、原始失败历史及负结果；不在这些开发观测上继续按个案追加门槛或挑子集宣布成功，后续若有不同机制须另行登记。
'''
 (EXP/'COMBINED_REPORT.md').write_text(report)
 dump(EXP/'COMBINED_DECISION.json',{'run':'completed','measurement':'R39_and_R40_independent_CPU_PASS','scope':'G_only_no_parameter_TTA_no_new3Dboxes','R39_original_center_preserved':True,'R40_standard_rectIoU_not_new_model_or_algorithm':True,'R39_bbox_center_mIoU_percent':s['balanced_mIoU_percent']['bbox_fullmap'],'R40_bbox_rectIoU_mIoU_percent':s['balanced_mIoU_percent']['bbox_IoU_fullmap'],'point_mIoU_percent':s['balanced_mIoU_percent']['point_compatible_fullmap'],'legacy_G_mIoU_percent':s['balanced_mIoU_percent']['legacy_G'],'current_configs_adopted_for_TTA':False,'no_default_replacement':True,'all_negatives_unknowns_retained':True,'A208_exposed_development':True,'no_U_T_FINAL_access':True,'identity_annotations_unchanged':True,'actual_R39_generations':209,'R40_new_generations':0,'lost_raw_generation_time_tokens_unknown':True,'audit_hashes':{'R39':sha(R39/'independent_audit/AUDIT_RESULT.json'),'R40':sha(HERE/'independent_audit/AUDIT_RESULT.json')},'identity_summary_sha256':sha(EXP/'IDENTITY_PUBLIC_SUMMARY.json')})
 print('COMBINED_R39_R40_REPORT',len(report))
if __name__=='__main__':main()
