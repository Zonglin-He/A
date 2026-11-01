"""Finalize saved R34/R35 evidence only: no inference, scoring, fitting, or network."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round35';EXP=HERE/'experiment';R34=ROOT/'research/round34'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 a34=read(R34/'independent_audit/U_AUDIT_RESULT.json');a35=read(HERE/'independent_audit/U_AUDIT_RESULT.json');assert a34['state']==a35['state']=='PASS'
 d=read(R34/'experiment/diagnostic_SUMMARY.json');u=read(EXP/'U128_SUMMARY.json');ad=read(EXP/'ADEV_SUMMARY.json');rr=read(EXP/'U128_ALL_ROWS_PRIVATE.json')
 qdiff=[{'alias':r['alias'],'platform':r['platform'],'R30_query':r['queries']['R30_U_rows'],'partial_query':r['queries']['partial_output_keep'],'R30_IoU':r['IoU']['R30_U_rows'],'partial_IoU':r['IoU']['partial_output_keep']} for r in rr if r['queries']['R30_U_rows']!=r['queries']['partial_output_keep']]
 assert len(qdiff)==4 and all(r['IoU']['R30_U_rows']==r['IoU']['partial_output_keep'] for r in rr)
 dump(EXP/'PARTIAL_R30_U_QUERY_DIFFERENCES.json',{'all128_IoU_exact':True,'different_query_count':4,'rows':qdiff,'note':'Different query does not establish different physical instance; equal IoU does not mean parameters or score vectors unchanged.'})
 pooled={k:(d['T128']['balanced_mIoU_percent'][k]+read(R34/'experiment/U128_SUMMARY.json')['U128']['balanced_mIoU_percent'][k])/2 for k in ['source','R30_ungated','h_scoped_R30','noun_scoped_R30']}
 dump(EXP/'T_U_POOLED_DESCRIPTIVE.json',{'n':256,'pool':'T128 plus U128 equally weighted within each platform','balanced_mIoU_percent':pooled,'h_minus_noun_pp':pooled['h_scoped_R30']-pooled['noun_scoped_R30'],'primary_selection_uses_pooled':False,'independent_confirmation':False,'reason':'T128 already exposed during h mechanism development; same historical scenes'})
 names={'source':'Frozen source','R30_ungated':'R30：无传播门','h_scoped_R30':'h门 R30（预定主方案）','noun_scoped_R30':'noun门 R30','h_samegate_QP':'同h门 QP，不更新','noun_samegate_QP':'同noun门 QP，不更新','h_global_mean':'h全局mean强度','h_global_norm':'h全局norm强度','noun_global_mean':'noun全局mean强度','noun_global_norm':'noun全局norm强度','R30_U_rows':'R30：原保持集合','U_rows_matched_denominator':'原集合＋匹配新分母','partial_output_keep':'新增未判断坐标保持'}
 diag='\n'.join('| '+names[k]+' | '+' | '.join(f"{d[s]['balanced_mIoU_percent'][k]:.6f}" for s in ['A208','DEV64','R64','S64','T128'])+' |' for k in d['DEV64']['balanced_mIoU_percent'])
 utable='\n'.join('| '+names.get(k,names.get(k.replace('R34_',''),k))+' | '+f"{u['drone'][k]['mIoU_percent']:.6f} | {u['quad'][k]['mIoU_percent']:.6f} | {v:.6f} | {v-u['balanced_mIoU_percent']['source']:+.6f} |" for k,v in u['balanced_mIoU_percent'].items())
 ttable='\n'.join('| '+names[k]+' | '+f"{ad['DEV64']['drone'][k]['mIoU_percent']:.6f} | {ad['DEV64']['quad'][k]['mIoU_percent']:.6f} | {ad['DEV64']['balanced_mIoU_percent'][k]:.6f} | {ad['A208']['balanced_mIoU_percent'][k]:.6f} |" for k in ad['DEV64']['balanced_mIoU_percent'])
 tailtable='\n'.join('| '+names.get(k,names.get(k.replace('R34_',''),k))+' | '+str(u['drone'][k]['improved'])+'/'+str(u['drone'][k]['worsened'])+' | '+str(u['quad'][k]['improved'])+'/'+str(u['quad'][k]['worsened'])+' | '+str(u['drone'][k]['query_changes'])+'/'+str(u['quad'][k]['query_changes'])+' |' for k in ['R30_U_rows','R34_h_scoped_R30','R34_noun_scoped_R30','R34_h_samegate_QP','U_rows_matched_denominator','partial_output_keep'])
 report=f'''# R34/R35：局部支持读出与部分输出保持的跨平台验证

**状态：完成。** 全部预测封存、CPU算术复核、指定范围完整模型闭环、封存后离线评分及独立U审计均已通过。主方法在U访问前由DEV固定为 **h_scoped_R30**；本报告不因U上R30更高而更换主方法。FINAL384未用于观测、拟合、方法选择或任务评分。

**结论：h局部读出在U128相对frozen提高0.520657pp，但相对原R30下降0.831878pp，未证明传播门比原R30稳定更好。新增未判断坐标的保持项确实降低相应输出漂移，却降低DEV任务指标，U逐例IoU也没有超过R30；停止该固定mask分支。** 两项实验都保留同信息不更新对照、强度/归一化控制及全部负比较，没有按GT删除22条反馈。

## 1. 本轮具体解决什么，保持什么

共同初始化是同一Waymo-only source checkpoint（SHA256 `0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`），学生输入仍为XYZ点云和文字。框坐标、候选池与非匹配网络冻结。监督沿用R29缓存的RGB＋完整文字Qwen偏好：22条偏好来自19个A观测，Drone10条/8观测，Quad12条/11观测。未知第三实例不自动变成负例。此前反馈的目标正确性和可迁移性局限继续存在；本轮没有新Qwen调用、没有新G、没有新几何回归。

R34只替换修正传播的检索key：source noun×query64改为归一化source h288（最终decoder query进入匹配投影之前，已由点云和文字条件化）。R30的已训练b1匹配MLP及全部22偏好不变，R34本身0个新优化步骤。借鉴GRACE的隐表示局部路由思路，**不是GRACE复现**，没有学习其值/半径或使用正确编辑oracle。前置key自排除诊断只观察近邻间隔；实际A读出不排除自身。

每个候选先找各A行内最近query，再做实际GPU float32 max/topk4，固定k=1。门只认原接受P/N端点；全反馈行支持是另一机制。完整GPU近邻索引、余弦、权重和门已保存，CPU按这些实际索引复核，不偷偷用CPU topk替换。两个key都配同信息、同key、同门的无参数QP记忆对照：对实例组做最小L2 margin0.1修正，每组只计一次，先检索原始修正再相对当前source q0锚定。全局mean/norm控制匹配局部门覆盖量或修正范数，区分局部路由与简单缩小步长。

R34推理坐标为 `ell_eps(p)=log(p+1e-6)-log(1+1e-6-p)`，对固定source q0作差，二值门0/1显式恢复source/R30端点。它保留官方float32饱和并列，**不等同于训练exact log-odds**。原R32 legacy float32跨实现近邻/并列警告仍是历史局限；本轮通过记录真实GPU近邻和新闭环缩小测量歧义，没有修改旧R32结果。

R35只改R30的保持mask：原来对没有反馈的189条A观测全部候选保持，现在另外保护19条反馈观测中未被接受P/N涉及的候选相对分数。仍为共享query匹配MLP184960参数、固定source q0、exact log-odds、softplus偏好温度0.1、beta1、AdamW lr5e-4/weight_decay0、16步、seed0；每平台独立source初始化。三配置×两平台共6状态、96优化步骤，没有超参搜索或U拟合。没有改adapter、追加RGB训练或使用标签筛选。

## 2. U之前的诊断和唯一主方法选择

单位均为mIoU%，最后一列与前几列有历史复用，不能相加当作独立证据。A用于拟合/保持；DEV已暴露并负责选择；R/S/T是历史机制诊断。

| R34方法 | A208 | DEV64 | R64 | S64 | T128 |
|---|---:|---:|---:|---:|---:|
{diag}

h门DEV25.379359，高于原R30的25.090904和noun门24.504612；因此在U数据访问前的统一锁中选为唯一主方案。历史R64退化、S64回到frozen、T128仅小幅提高，开发排名并非跨集合稳定优势。原R30、两种局部门和两种QP都实现6/6要求换目标的方向及6/6最终正组；四个全局缩步对照为0/6，不能把回到frozen当作保留纠错后的保护成功。反馈落实不等于反馈GT正确。

R35开发比较如下。归一化控制保持原189行分子，但分母换为partial新分母，避免把平均项变弱误当mask的作用。

| R35方法 | DEV Drone | DEV Quad | DEV等权平均 | A208等权平均 |
|---|---:|---:|---:|---:|
{ttable}

partial比R30低2.014808pp，比匹配分母控制仍低0.584461pp。三配置都保留6/6换目标方向/最终正组，差异不是监督完全未执行。Drone/Quad原保持坐标24576/23808，新mask26042/26484，新增未判断坐标1466/2676，释放端点582/140。匹配分母控制相当于原保持项名义beta乘0.943706/0.898958；这不是梯度范数或Adam实际步幅匹配。全部22偏好、分母和更新预算固定。

独立CPU64复核新增受保护区域MSE：Drone R30 3.291900→partial2.473765；Quad0.331000→0.287816（匹配分母控制3.439175/0.353520）。保持机制数值上生效，但其减少的漂移没有带来任务收益。不能据此推断所有输出保持或局部适应均无效。

## 3. U128：固定方法的新内容确认

U为Drone64＋Quad64，在RGB解码像素与canonical XYZ实际内容上与历史池和T128去重；独立内容审计通过。它仍来自1个Drone、3个Quad历史场景，且此前存在官方全验证集暴露。**这是新观测内容的固定方法确认，不是新场景、从未暴露目标域或严格场景独立测试。** 平台等权平均为两平台各自均值再平均。

| 方法 | Drone | Quad | 等权平均 | 相对source pp |
|---|---:|---:|---:|---:|
{utable}

预定主方案h门相对source为Drone+0.034481pp、Quad+1.006832pp、等权+0.520657pp；相对同信息同门QP的等权增量+0.418020pp，其中Drone反而−0.170791pp，Quad+1.006832pp。相对noun门仅+0.047386pp；相对原R30为−0.831878pp。原R30在U更高是固定对照的观察结果，不是新的主方法选择依据。

| 方法 | Drone改善/恶化 | Quad改善/恶化 | Drone/Quad换query数 |
|---|---:|---:|---:|
{tailtable}

h门以少量改动保留3条改善、1条恶化，但也丢掉原R30多条有益变化；低覆盖不能自动等同于正确选择了适用区域。四个全局强度控制U任务IoU等于source，但部分query确有变化，不能写成连续分数、参数或所有预测不变。

partial与R30的全部128条IoU逐例完全相同，包括所有负例；其中 **4条query不同**，明细单独导出。由于候选框重复/均为零IoU等原因，任务相等不等于query、参数或分数相等，也不能凭query不同认定物理实例不同。partial相对匹配分母控制U+0.204153pp，但相对其应替代的R30为0，且DEV更差，因此不采用此mask。

全部平台Acc@25/50、逐例IoU、query变化、场景表、成对比较与负比较另存机器可读文件。不能只报告平均正收益。本轮没有以任何负例IoU作为在线门或删反馈依据。

补充总体描述：把T128和U128在各平台等量合并为256条，source/R30/h/noun等权mIoU分别为{pooled['source']:.9f}/{pooled['R30_ungated']:.9f}/{pooled['h_scoped_R30']:.9f}/{pooled['noun_scoped_R30']:.9f}。h比noun仅{pooled['h_scoped_R30']-pooled['noun_scoped_R30']:+.9f}pp，原R30比source{pooled['R30_ungated']-pooled['source']:+.9f}pp，h比source{pooled['h_scoped_R30']-pooled['source']:+.9f}pp；DEV较优没有变成总体key优势。T已经参与h机制开发，该合并表只作描述，不称256条独立确认，不用它改变U主方案，也不加不可估的场景CI。

## 4. 观测簇、不确定性和时间边界

U的Drone仅1场景，无法估计跨Drone场景泛化区间；两平台总体场景置信区间不可估。保存的场景条件重采样分位数只是条件描述，不是总体95%CI或显著性证据。h比source的条件分位数虽为[0.017241,2.165149]pp，也不改变这一边界。全部场景指标保留共享场景匿名标识，不能把64帧当64个独立场景。

U确认前的统一锁固定h和所有控制。更早“今晚最终只允许四方法”的表述在**U评分前**已被单独澄清：这是第一阶段提名，不禁止用户已授权的后续独立登记机制；本轮h主方法不变。U评分后已暴露，只能再作后续诊断/开发，不能称下一轮独立确认。FINAL384仍完全未用，任何最终比较必须另立不可变方法/控制锁；本报告不宣布已完成FINAL，也不混入未执行建议。

## 5. 独立审计和真实执行量

R34诊断528×10=5280个任务值和完整读出分数复核，19组QP独立求解、220条偏好执行记录保留；U128×10=1280个分数/任务值复核，完整分数最大误差0。R35独立重放6个保存头并检查512个U概率/exact log-odds向量，最大误差分别1.18574e-6/6.70450e-6，所有CPU argmax一致；source/R30跨两路概率逐位相同。全部U128选中query和框、mask和漂移分解独立复核。

实际GPU完整模型闭环：R34诊断按事先散列规则每split×platform取1条，共10条/20前向；R34 U全部128条source＋R30共256前向；R35 U两新头共256前向，source/R30复用R34。**U共有512次不重复完整模型闭环前向，另加诊断20次；缓存匹配模块CPU/GPU重放不是额外完整backbone前向。** R34两个闭环各耗时3.2224s/12.5905s；R35六状态拟合4.6227s，U匹配重放2.9908s。这些是指定阶段回执，既不含全部采集/审计/数据准备，也不是优化后的端到端单方法部署延迟。

独立U审计最初误用“所有闭环连续分数逐位相同”的假设，失败日志保留；随后按原闭环回执允许的误差核对，最大坐标分数差7.45301e-6，最终query/框精确一致，没有改动预测。不能隐藏最初审计失败，也不能将修复审计口径描述成修改结果。

导出含R34诊断599、R34 U204、R35 A/DEV135、R35合并U400条负比较；它们来自不同方法/参考对且有复用，不是1338个独立负样本。R34共6560条臂任务标量；R35共2624条（含共享source/R30与R34控制复录），重复部分显式保留，不当成新增证据。原始媒体、文字、物理坐标、opaque ID、特征/权重和第三方源码不导出。

## 6. 裁决

- **R34：保留其固定主方案身份和全部控制，但不认定它优于R30或已经稳定解决负迁移。** 它在U以局部小收益超过frozen/QP，却低于原R30；历史集合也不一致。
- **R35：不采用并停止当前partial-mask分支。** 保持项数值有效，DEV变差，U对R30无额外任务收益；不追加强度搜索以追U排名。
- **不通过U挑赢家、删反馈或自动追加第三候选负标签。** 保留全部22反馈、所有负例、相同信息不更新对照和封存审计；FINAL另锁后才能进入最终确认。
'''
 (EXP/'REPORT.md').write_text(report)
 decision={'run':'completed','measurement':'independent_U_audits_PASS','fixed_primary':'h_scoped_R30','primary_chosen_before_U':True,'post_U_reselection':False,'R34_adopted':False,'R34_verdict':'small U gain vs frozen and samegateQP, lower than ungated R30; no stable superiority established','R35_adopted':False,'R35_stop_branch':'fixed partial keep mask, no new strength sweep','R35_mechanism_effect':'new unjudged coordinate MSE reduced','R35_task_effect':'DEV below R30, all128 U IoU exact R30 with4differentqueries','all22_feedback_retained':True,'new_G_calls':0,'R34_new_optimizer_steps':0,'R35_optimizer_steps':96,'U_primary_locked_before_access':True,'U_now_exposed_for_future_diagnostics':True,'FINAL384_manifest_media_GT_not_read':True,'FINAL_requires_separate_immutable_lock':True,'night_continuation_not_prohibited_by_this_closeout':True,'publication':'local_only_pending_parent','U_balanced_mIoU_percent':u['balanced_mIoU_percent'],'audit_hashes':{'R34':sha(R34/'independent_audit/U_AUDIT_RESULT.json'),'R35':sha(HERE/'independent_audit/U_AUDIT_RESULT.json')}}
 dump(EXP/'DECISION.json',decision);dump(EXP/'RUN_STATE.json',{'state':'COMPLETED_INDEPENDENTLY_AUDITED_LOCAL_ONLY','R34diagnostic_observations':528,'U_observations':128,'R34_fullclosure_forwards':276,'R35_new_fullclosure_forwards':256,'same_source_R30_not_doublecounted':True,'no_GPU_scoring_or_training_by_finalizer':True,'rootSTATE_not_modified':True})
 paths=list((ROOT/'crossplatform/round34').glob('*.py'))+list((ROOT/'crossplatform/round35').glob('*.py'))+list((R34/'independent_audit').glob('*.py'))+list((R34/'key_diagnostic').glob('*.py'))+list((HERE/'cpu_checks').glob('*.py'))+[Path(__file__),HERE/'publish35_combined.py']
 dump(EXP/'FINAL_CODE_SNAPSHOT.json',{'scope':'own code only, no thirdparty snapshots','files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(paths)]})
 print('REPORT35_COMBINED_WRITTEN',len(report),len(qdiff))
if __name__=='__main__':main()
