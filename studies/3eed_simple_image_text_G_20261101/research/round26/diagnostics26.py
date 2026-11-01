"""Post-seal descriptive ablations; no retuning or model execution."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round26'))
from common26 import *
def main():
 mx=0.;saturated=0;scorechanged=0;qchanged=0
 for r in rows():
  z=load_npz(BASE/'scores464_attempt001/samples_PRIVATE'/(r['id']+'.npz'));nn=z['cue_names'].tolist();a=z['cues'][nn.index('no_position_reference')];b=z['cues'][nn.index('centered_unclipped_gray_dual')];mx=max(mx,float(abs(b).max()));saturated+=int((abs(b)>2).sum());scorechanged+=int(not np.array_equal(a,b));qchanged+=int(np.argmax(z['source']+2*a)!=np.argmax(z['source']+2*b))
 dump(EXP/'CLIPPING_DIAGNOSTIC.json',{'unclipped_max_abs_cue':mx,'saturated_query_count':saturated,'cases_with_score_difference':scorechanged,'argmax_changes':qchanged,'scope':'all464samepool,zeroGTusedfornumericclippingaudit'})
 neg=read(EXP/'ALL_NEGATIVE_CASES.json');dump(EXP/'NEGATIVE_DIAGNOSTIC.json',{'negative_comparison_records':len(neg),'unique_observations_with_any_negative_comparison':len({x['alias'] for x in neg}),'negative_comparisons_with_unmapped_new_choice':sum(not x['mapped_new'] for x in neg),'interpretation':'postseal descriptive; repeated arms/refpairs not independent'})
 ef=read(EXP/'EFFECTS.json');p=read(EXP/'DECISION.json');p.update(fulltext_scene_variant_prespecified_but_not_D208_selected={'P64_gain_vs_legacy_pp':ef['P64__qwen_full_scene_raw__vs__legacy_complex']['balanced']['gain_pp'],'OLD400_gain_vs_legacy_pp':ef['OLD400__qwen_full_scene_raw__vs__legacy_complex']['balanced']['gain_pp'],'not_switched_based_on_P64':True},only_position_reference_combined_increment_tested=True,lambda_search_boundary_limited=True,clip_rule_no_argmax_effect_in464=qchanged==0);dump(EXP/'DECISION.json',p)
 marker='\n\n## 删项细节与未证实解释'
 report=(EXP/'REPORT.md').read_text().split(marker)[0]
 report+=marker+f'''\n\n位置与参照一起删除，只证实合并增量，不能把收益全部归给独立参照定位。取消残差裁剪影响{scorechanged}条观测、{saturated}个query分数，Top1变化{qchanged}；仅是本批决策无影响，不是全局数学等价。去灰图在去位置/参照条件下，OLD400 +0.310523 pp、P64 0；不能推断恢复关系评分后相同。单context相对双裁剪的bounded版本，OLD400 +0.317606 pp、P64 -0.109405 pp，缺乏稳定必需性。raw双裁剪去灰图在P64损失3.025515 pp，作用依赖读出形式。\n\n完整文字整图单分数P64相对旧G +0.776290 pp（5条改善、4条恶化），OLD400 -2.572621 pp、REPLAY192 -2.572277 pp；是预定低调用量候选，未按P64改选主方案。D208选中的是单context主体raw，P64 -2.665839 pp。共享λ只查预定有限网格，上界最佳不等于全局调参最优，未在P64后追加扫描。\n\n有pool时源q0在本批均有映射，不能把下降直接归为q0未映射。绝对分数、未映射候选与source评分交互为后封存描述，未建立单一因果归因。全文整图与主体裁剪还存在分辨率/上下文差异，不能把失败单独归因于关系推理。\n'''
 (EXP/'REPORT.md').write_text(report)
 print('POSTSEAL26',scorechanged,saturated,qchanged,len(neg))
if __name__=='__main__':main()
