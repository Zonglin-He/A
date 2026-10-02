"""Write the measured decision with uncertainty and interface limits intact."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_transfer_aligned_common_v1 import *
def estimate(m,scale=100,digits=4):
 return f"{m['mean']*scale:+.{digits}f} [{m['ci95'][0]*scale:+.{digits}f}, {m['ci95'][1]*scale:+.{digits}f}]"
def run():
 ts=read(PUBLIC/'hc2/TRANSFER_SUMMARY.json');cc=ts['corruption'];tr=read(PUBLIC/'TRANSFER_RESOURCES.json');tt=read(PUBLIC/'TOKEN_RESOURCES.json');a=read(PUBLIC/'TRANSFER_ROOT_READBACK.json');b=read(PUBLIC/'TOKEN_ROOT_READBACK.json');pa=read(BASE/'PUBLIC_AUDIT.json')
 lines=['# Single-write transfer is possible; text-nearest advantage and token qualification remain limited','',
 '2026-10-02. 两项用户授权的机制实验均已实际完成并在各自预测/分数封存后做GT评分。HC单次历史写入能够跨query产生正效用；没有出现“near平均正、far平均负”的分离。R的near均值高于far，但配对区间跨零，因此没有确认语义距离是决定适用性的规则，也没有把prototype/memory接入方法。冻结CLIP逐token/patch版本改善了表示分离，却没有建立对Sa2VA的排序或互补优势。本轮未跑新的online方法、fusion、memory或Vid routed stream。','',
 '## HC Single-Write：比较的是保存状态对，不是Frozen主榜','',
 '原HC32历史曝光来源、一query/源、双序、clean+五类transient5%corruption。两序scheduled donor并集14来源，每条件/臂16个写入，含空证据与no-op：A/R各96，共192保存pre/post状态对。每状态对测self/next/near/far四角色，768逻辑pair；角色重复时只推理一次，保留角色读出。共1464 cached suffix replay、32个冻结text-only RoBERTa前向，零新backbone/专家/backward/训练更新。','',
 'z是HC原checkpoint的RoBERTa body最终lexical hidden mean，排除special/padding并L2-normalize；不是视觉相关post-MM H，也不是新训练的context encoder。32x32cosine和目标表在新GT读取前锁定。next为下一条nonexpert；near/far在原order所有未来不同query中选极值，ties选最早位置。A/R/各condition用同一目标。未来near/far可原本scheduled，但本次不调用目标专家或更新目标状态。','',
 'Tij=M(xj;历史ψi+)−M(xj;历史ψi−)，没有把Δ搬到ψ0。主空间比较固定target的checkpoint native interval，A/R/pre/post完全相同；self另按旧pre-native interval精确复现局部结果，本轮所有self该interval与共同冻结interval相同。Dense sIoU覆盖全部合法GT帧；vIoU使用原物理时间及官方HC clipping。Free native解码另存，本轮所有pair的ΔtIoU=0。','',
 '下表是corrupt子集每角色80pair/臂，源宏平均，95%CI为10000 donor-source bootstrap（seed20261001）。单位pp；不是与Frozen主输出的在线增益。','',
 '| Target | A Uniform一次写入Δv | R Routed一次写入Δv | R−A |','|---|---:|---:|---:|']
 for role,label in [('self','当前query（self control）'),('next','下一条nonexpert'),('near','文本最相似future'),('far','文本最不相似future')]:lines.append(f"| {label} | {estimate(cc[role]['A']['metrics']['delta_v'])} | {estimate(cc[role]['R']['metrics']['delta_v'])} | {estimate(cc[role]['R_minus_A']['metrics']['delta_R_A_v'])} |")
 lines+=['',f"R near−far = {estimate(cc['near_minus_far']['R']['metrics']['delta_near_far_v'])}pp；A对应 {estimate(cc['near_minus_far']['A']['metrics']['delta_near_far_v'])}pp。R两个order的near−far为+{cc['near_minus_far']['R']['metrics']['delta_near_far_v']['order_values'][0]*100:.4f}/{cc['near_minus_far']['R']['metrics']['delta_near_far_v']['order_values'][1]*100:+.4f}pp，不能把正均值写成两个order均成立。near/far各只有8不同target来源，next13；全部donor14聚类。target聚类敏感性、clean、sIoU、实际write/no-op、gross gain/loss与负例均完整公开。",'',
 '解释边界：R的self和所选future目标均值可正，削弱“空间更新只能修当前query”的解释。R near对A的配对差为正区间，但near−far未排除零，far本身为正，不符合简单的near正/far负故事。条件化context可作为候选解释，不能据此确立记忆路由规则。历史基态A/R不同，差异同时包含各自更新和已有状态，不能把R−A唯一归因为更generic或更specific。单写入为正而旧完整online R−A未赢，与累积/交互可能性相容，也没有reset/matched-base顺序消融来唯一确认它。','',
 'Corrupt A实际write75/80、R77/80；self正/负/零A50/20/10，R66/6/8。R near69/3/8，far48/24/8；单写入四角色均无>5pp的vIoU负尾。这个局部预算下的结果不抵消旧online的严重负尾，也不证明继续多次写入安全。gross表按cell计，不能直接与不均匀donor-source macro相减。','',
 '## CLIP Token P1：显式词组、多token、多patch，无online更新','',
 '复用旧60cell/20来源/540条A首步固定tube、原共同native interval、已有routed Sa2VA rewards。各集10clean+20corrupt，固定候选来自原历史A状态，不是新的Frozen checkpoint候选。冻结OpenAI CLIP ViT-B/16（官方HF revision57c216476eefef5ab752ec549e440a49ae4ae5f3），FP32/eager，最终patch post-LN+visual_projection与lexical text final-LN+text_projection共同512维空间。显式Stanza依存/POS解析object/event phrase，逐phrase编码后保留lexical BPE tokens，未把整个句子或branch pool成一个向量。','',
 '所有原采样帧及完全相同corrupted pixels逐clip SHA匹配；full-frame224x224 bicubic，无center crop。Candidate ROI为14x14patch中正面积交叠集合。T=event token各自max-inside−max-outside，再token平均；object max只解释，不与T融合。原CLIP CLS/EOT投影逐位parity通过。','',
 '这不是FILIP：CLIP预训练监督的是pooled图文特征，投影相同不保证lexical/patch局部语义已经正确校准。本轮检验的是这个零训练局部读出。FILIP的token-wise预训练目标、MaskCLIP的dense计算改写都没有在这里假装复现。静态逐frame编码不建模运动，非等大的in/out最大池有极值偏置，224patch硬ROI对小参数probes可能没有分辨率；这些界限保留，不做本轮权重/温度/候选/解析规则扫参。','',
 '| Dataset | T pairwise，合法strict-pair子集 | T−candidate0 top1，完整corrupt panel | T−Sa2VA top1，完整panel |','|---|---:|---:|---:|']
 for ds,label in [('hc2','HC'),('vidstg','Vid')]:
  m=read(PUBLIC/ds/'TOKEN_SUMMARY.json')['corruption']['metrics'];lines.append(f"| {label} | {estimate(m['T_pairwise'])}% | {estimate(m['T_gain'])}pp | {estimate(m['delta_T_S_v'])}pp |")
 lines+=['',
 'HC全部20corrupt cell/10来源可评分，720 GT-strict pairs；同子集S=81.6667%，T=52.2917%，T的CI跨50%。Vid10/20corrupt cell有event tokens，其余10保留unavailable+native fallback；10合法cell中4个GT九候选全并列，所以T pairwise只用6cell/3来源，S相同子集80.0926%、T53.7037%，区间很宽。完整16个有strict GT pair的cell计空event半信用fallback，VidT=51.3889% [45.3125,58.8542]。全部20corrupt top1包括native fallback，完整面板和合法子集都报告，避免悄悄删无event输入。','',
 '原native版本pooled q_obj/q_evt cosine接近.99；本版lexical-token均值的诊断cosine HC=.68095、Vid=.56228（仅合法event输入）。这个方向差异不能单独当作grounding正确性。五个Vid query解析后没有lexical VERB/root是NOUN，按预锁规则event为空；不从GT决定补全。','',
 '关键新断点在分数分辨率：HC20corrupt cell里8个九候选全部同分，9个只有两个唯一T分；平均1.9种T分，虽然平均8.9种ROI signature，表明很多ROI改变没有改变winning maxima。Vid10个可评分cell中4个全同分，另外10不可评分。GT-strict pair内HC T ties545/720；Vid合法T ties155/216。S错T对/S对T错HC28/69（545ties另列）、Vid0/31（155ties与360不可测pairs另列）。因此不能把减少绝对坏pair数当作互补改善：很多pair转成了tie或unavailable。','',
 '本轮没有建立token优于Sa2VA或足够可靠的互补优势，不做fusion/online接入。HC T的正小均值与wide CI保留，不把它叫已证无效；Vid可评分规模很小，且lexical local alignment、静态event、硬patch max存在接口边界，不能普遍否定token-level binding。事件外对象标签仍缺失，没有虚构right-object/wrong-event准确率。','',
 '## 资源、根核验与保存工程记录','',
 f"Transfer成功worker wall={tr['worker_wall_seconds']:.2f}s；CLIP成功worker wall={tt['worker_wall_seconds']:.2f}s，{tt['frames']}帧。均含load/IO，token还含decode/CPU评分，不是pure GPU kernel时间。Transfer peak allocated{tr['peak_vram_bytes']/2**30:.2f}GiB；CLIP peak{tt['peak_vram_bytes']/2**30:.2f}GiB。CLIP602356502bytes官方资产/LFS SHA校验，权重不公开。无新Sa2VA/训练/optimizer，CURRENT_METHOD与旧队列不改。",'',
 f"根transfer核验{a['checks']}，maxmetricerror={a['max_metric_error']:.3g}；192 self pre/post boxes/indices逐位复现，96配对局部标量复现旧报告。Token{b['checks']}，全540候选独立ROI坐标/max/cosine复算，maxerror={b['max_error']:.3g}；全候选GT utility及S reward精确复现旧资格。公开derived values/target extremum/source-bootstrap复核{pa['checks']}项通过。",'',
 '启动与评分的两个工程记录原件保留：pre-launch CUDA wrapper没有executable bit，显式bash调用，0预测；CPU closing checker误以为旧paired文件有A_net/R_net字段，实际只有R−A标量，改用配对差核对。第二项发生在全预测seal后GT已读，原metric rows逐值保持，没有重选目标、参数、候选或改推理。另一次根收尾元数据检查误把目标锁的selection_time读成time；修正字段后重新核验封存时序，未读取新GT或修改测量。收尾时只写一次JSON保护也拒绝了已有文件的重复写入，改为核对已有审计与单列收尾记录，原件保留。详见ENGINEERING_RECOVERIES.json和CLOSING_ENGINEERING_RECOVERIES.json。','',
 '## 本轮决策','',
 '两项资格实验已完成；保留原A研究对照和HC routed的局部正证据。HC不是全部cross-query失效，语义near−far规则仍未定，accumulation/interference未唯一确认。Token当前max读出与coverage不足，暂不接方法。本轮不自动实现memory、anchor/replay、temporal多视图、ST融合或新的online集成实验。新方案应据这轮具体断点再做独立匹配检验，不能把附件的最终pipeline当成已经执行的系统。','',
 '[CLIP官方代码](https://github.com/openai/CLIP/blob/main/clip/model.py)、[官方checkpoint](https://huggingface.co/openai/clip-vit-base-patch16)、[FILIP原论文](https://arxiv.org/abs/2111.07783)、[MaskCLIP ECCV22官方](https://github.com/chongzhou96/MaskCLIP)。这里只引用共同投影/训练目标/局部接口区别，不把它们的task结果转移成我们的成绩。','',
 '![Measured comparison](TRANSFER_ALIGNED.png)','',
 '![Selected transfer pairs](SELECTED_TRANSFER_MATRIX.png)','',
 '图包含corrupt源聚类95%CI；token pairwise明确同合法strict-pair子集。矩阵只画预选future pair，灰色未测，重复角色只计一次，不是全N²。PNG/PDF/SVG皆提供。']
 text='\n'.join(lines)+'\n'
 audit_path=PUBLIC/'PUBLIC_AUDIT.json'
 if audit_path.exists():assert read(audit_path)==pa,'saved public audit changed'
 else:write(audit_path,pa)
 (PUBLIC/'REPORT.md').write_text(text)
 doc=text.replace('(TRANSFER_ALIGNED.png)','(../results/tastvg_transfer_aligned/2026-10-02/TRANSFER_ALIGNED.png)').replace('(SELECTED_TRANSFER_MATRIX.png)','(../results/tastvg_transfer_aligned/2026-10-02/SELECTED_TRANSFER_MATRIX.png)');(ROOT/'docs/TA_SINGLE_WRITE_ALIGNED_TOKEN_REVIEW.md').write_text(doc)
 status(BASE/'STATUS.json',dict(status='completed_pending_verified_publication',transfer_pairs=768,token_cells=60,time=time.time()))
if __name__=='__main__':run()
