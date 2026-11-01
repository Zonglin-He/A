"""Summarize finite real-input controls; no further tuning or parameter adaptation."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round27'))
from common27 import *
def main():
 assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';effects=read(EXP/'EFFECTS.json');sel=read(EXP/'SELECTED_LAMBDAS_LOCK.json')['lambda_by_variant'];names=['source','legacy_complex','R26_scene_fixed2','R26_scene','scene_base','scene_hires','scene_repeat','crop_scene_joint'];summ={p:read(EXP/(p+'_SUMMARY.json')) for p in ['OLD400','REPLAY256','OLD464','Q64']}
 def effect(p,a,b):return effects[p+'__'+a+'__vs__'+b]['balanced']
 contrasts={}
 for p in ['OLD400','REPLAY256','OLD464','Q64']:
  for b in ['source','legacy_complex','R26_scene_fixed2','scene_base','scene_hires','scene_repeat']:contrasts[p+'__joint_vs_'+b]=effect(p,'crop_scene_joint',b)
 q=contrasts['Q64__joint_vs_legacy_complex']['gain_pp'];old=contrasts['OLD400__joint_vs_legacy_complex']['gain_pp'];adopt=q>0 and old>0
 signal=all(contrasts[p+'__joint_vs_'+b]['gain_pp']>0 for p in ['OLD400','Q64'] for b in ['source','R26_scene_fixed2'])
 decision={'state':'completed_measured_audited','primary':'crop_scene_joint','lambda':sel['crop_scene_joint'],'Q64_joint_vs_sameprompt_scene':contrasts['Q64__joint_vs_scene_base'],'Q64_joint_vs_legacyG':contrasts['Q64__joint_vs_legacy_complex'],'OLD400_joint_vs_legacyG':contrasts['OLD400__joint_vs_legacy_complex'],'OLD400_joint_vs_R26scene':contrasts['OLD400__joint_vs_R26_scene_fixed2'],'supervision_positive_as_bounded_research_signal':signal,'replace_research_precision_reference':adopt,'robust_cross_platform_advantage_established':False,'full_regression_attributed_to_resize':False,'all_negative_records_retained':True,'no_Q64_driven_family_or_lambda_switch':True,'new_source_training_steps':0,'target_TTA_steps':0,'new_GPU_queued':False}
 dump(EXP/'DECISION.json',decision);dump(EXP/'KEY_CONTRASTS.json',contrasts)
 cost={n:{'calls':0,'images':0,'input_tokens':0,'visual_tokens':0,'model_forward_seconds':0.} for n in ['scene_base','scene_hires','scene_repeat','crop_scene_joint']};coverage={}
 for r in rows():
  e=read(BASE/'joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'));coverage.setdefault(r['stage']+'_'+r['platform'],{'n':0,'with_pool':0});coverage[r['stage']+'_'+r['platform']]['n']+=1;coverage[r['stage']+'_'+r['platform']]['with_pool']+=bool(e['targets_PRIVATE'])
  for m,ev in e['targets_PRIVATE'].items():
   for n in cost:
    v=ev[n];c=cost[n];c['calls']+=1;c['images']+=len(v['image_grid_thw']);c['input_tokens']+=v['input_tokens'];c['visual_tokens']+=sum(g['visual_tokens'] for g in v['view_geometry']);c['model_forward_seconds']+=v['forward_seconds']
 dump(EXP/'COSTS.json',{'same528_observations':cost,'model_forward_seconds_excludes_preprocessing_and_loading':True,'total_joint_stage':{k:v for k,v in read(BASE/'joint528_attempt001/SEALED.json').items() if not isinstance(v,(dict,list))},'target_TTA_steps':0});dump(EXP/'COVERAGE_SUMMARY.json',coverage)
 resolution=read(EXP/'RESOLUTION_SUMMARY.json');g=resolution['by_platform_and_group'];d=g['drone_legacy_winner_R26_harm'];quad=g['quad_legacy_winner_R26_harm']
 lines=['# R27：完整文字的主体裁剪＋场景联合G','',f"实际完成528条真实观测（旧464＋新Q64），同一Waymo-only100/256候选框及2D3D映射。四个新读出共用同一完整描述、同一提示词，只输出联合Yes/No logodds，F=source+λ*weights@e。没有手工位置、参照、灰图、残差分，没有参数TTA。主版本固定crop_scene_joint，D208只选共享λ={sel['crop_scene_joint']}，没有按Q64改选族。",'','## 先检查缩放假设','',f"旧400中R26整图相对旧G退化的可映射候选：Drone {d['n_masks_not_independent_samples']}条，缩放后框短边中位数{d['scene_short_px']['median']:.2f}px；Quad {quad['n_masks_not_independent_samples']}条，中位数{quad['scene_short_px']['median']:.2f}px。两组均无短边<28px样本。这不支持主体整体极小是普遍主因；不排除衣着等细节受影响。框尺寸不证明目标身份，放大不新增原始相机细节。",'','## 受控输入','','- scene_base：新统一提示词，整图，每图最多1024视觉token。与R26旧提示词的整图单分数并列，避免把提示词变化算成裁剪收益。','- scene_hires：相同提示词及图像，提高整图上限到2048 token。实际分辨率与token逐条记录，并非每图恰好翻倍。','- scene_repeat：同一标记整图重复两次，每图同1024上限，用于检查重复观测/多图预算的影响。','- crop_scene_joint：原图中目标周围20%上下文裁剪＋完整场景，两张图同时输入一次Qwen，二者均标记同一候选，产生一个分数。每图同1024上限；没有分别打分再加权。','- Frozen source、旧复杂G、R26整图固定λ2、R26同预算重新校准λ，以及每个新读出的fixed2版本均保留。所有版本共用候选池和框；未用GT挑目标、mask或运行时门。','','## 结果','','mIoU为%，均值为两平台等权平均；Acc25/50、场景记录、全部负例另附。']
 for p in ['OLD400','REPLAY256','Q64']:
  lines+=['',f'### {p}','','| 方案 | λ | Drone | Quad | 均值 |','|---|---:|---:|---:|---:|']
  for n in names:
   l='—' if n=='source' else '2固定' if n.endswith('_fixed2') or n=='legacy_complex' else str(sel[n]);dv=summ[p]['drone'][n]['mIoU_percent'];qv=summ[p]['quad'][n]['mIoU_percent'];lines.append(f'| {n} | {l} | {dv:.6f} | {qv:.6f} | {(dv+qv)/2:.6f} |')
 lines+=['','## 主要差值','','| 数据 | joint−同提示词整图 | joint−高分辨整图 | joint−重复整图 | joint−R26整图固定2 | joint−旧G |','|---|---:|---:|---:|---:|---:|']
 for p in ['OLD400','REPLAY256','Q64']:lines.append('| '+p+' | '+' | '.join(f"{contrasts[p+'__joint_vs_'+b]['gain_pp']:+.6f}" for b in ['scene_base','scene_hires','scene_repeat','R26_scene_fixed2','legacy_complex'])+' |')
 lines+=['','## 裁决和边界','',f"联合裁剪版本相对source和R26整图在旧400及Q64均值均为正：{signal}。联合版本是否替换研究精度参照：{adopt}。Q64相对旧G {q:+.6f} pp，场景聚类描述性95%区间 {contrasts['Q64__joint_vs_legacy_complex']['cluster95_pp']}；相对同提示词整图 {contrasts['Q64__joint_vs_scene_base']['gain_pp']:+.6f} pp。所有结果按总体净指标判断，不以单个负例否决。",'','像素统计本身不是可识别性认证；提高整图分辨率是输入分辨率的单因素干预，裁剪联合与重复整图的比较进一步检查局部放大是否有额外作用。多图的实际token预算不同，逐条保留，不能把任何联合输入收益全部归因于分辨率。没有把全部旧400退化归结为缩放。','','D208反复暴露目标标签开发，只选择一个两平台共享λ；有限网格增加4端点源于R26最佳常在2上界，所有新读出及R26重校准使用相同预算。所有528预测及参数选择锁定后才读本轮Q64评分标签。Q64图像解码/XYZ内容不重叠近期开发（含P64），但全VAL历史评分及2Drone/3Quad场景重叠；不称完全盲测或新场景。原400及重放256均已暴露。','','CPU审计验证旧464的source/旧G/R26整图固定2精确重放、全部528源候选框不变、所有选参和IoU独立复算；实际图像网格、提示词、完整文字、两张图的目标对应与分辨率静态审计逐条核对。全部变化/负例保留，多臂比较不是独立样本。','','本轮只验证G及其同信息直接重排。没有训练新模块或接入参数TTA，不能从这张表声称参数适应的额外收益。此前R24匹配TTA与R25源域融合保留。源权重仍为Waymo-only。','','实现遵循[Qwen2.5-VL官方多图和像素预算接口](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。资产固定为本地Qwen2.5-VL，未跟随旧GitHub链接重定向升级到Qwen3。私有诊断图只用于检查标记/裁剪，无原始媒体公开发布。初次CPU分辨率脚本遇到缺失OLD400清单，修复清单别名后所有推理输入哈希不变，失败日志保留。']
 (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n');dump(EXP/'RUN_STATE.json',{'run':'completed','measurement':'valid_CPU_audited','claim':'bounded_jointG_supervision_evaluation_no_TTA','decision':decision,'queued_GPU':False});print('FINAL27',decision)
if __name__=='__main__':main()
