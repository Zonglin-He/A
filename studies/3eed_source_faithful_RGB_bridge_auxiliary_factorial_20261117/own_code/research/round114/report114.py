from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round114';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;aa=rd(H/'scores_attempt001/AGGREGATE_ROWS.json');rows=[r for p in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];p=rd(H/'DECISION.json')['primary'];cases=[];conc=[]
 for c in ['sourcekey_frozen_fusedABC','neither_correct_fusedABC','anchor_bridge_natural_raw_fusedABC']:
  ds=np.array([r['arms'][p]['IoU']-r['arms'][c]['IoU'] for r in rows]);conc.append({'control':c,'n':len(ds),'gain_pp':float(ds.mean()*100),'better':int((ds>1e-12).sum()),'worse':int((ds< -1e-12).sum()),'same':int((abs(ds)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,ds.max())*100/768),'largest_negative_single_input_pp':float(min(0,ds.min())*100/768)})
  for name in rd(R/'crossplatform/round114/MATRIX_SEALED.json')['directions']:
   for label,sgn in [('good',1),('failure',-1)]:
    rr=[r for r in rows if r['source']+'_to_'+r['target']==name and not r['protected_alias_fallback'] and sgn*(r['arms'][p]['IoU']-r['arms'][c]['IoU'])>1e-12]
    if rr:
     r=max(rr,key=lambda r:abs(r['arms'][p]['IoU']-r['arms'][c]['IoU']));cases.append({'direction':name,'id_PRIVATE':r['id_PRIVATE'],'label':label,'control':c,'delta_IoU':r['arms'][p]['IoU']-r['arms'][c]['IoU'],'identity':'UNKNOWN_no_new_media_review','postseal_only':True})
 exact=[]
 for v in ['A','C','fusedABC']:
  a='native_gate_only_'+v;b='anchor_bridge_natural_raw_'+v;equal=sum(r['arms'][a]['query_PRIVATE']==r['arms'][b]['query_PRIVATE'] for r in rows);same=sum(r['arms']['native_bridge_no_gate_'+v]['query_PRIVATE']==r['arms'][{'A':'neither_correct_A','C':'anchor_frozen_C','fusedABC':'neither_correct_fusedABC'}[v]]['query_PRIVATE'] for r in rows);exact.append({'view':v,'n':768,'gateonly_vs_fullbridge_same_query':equal,'nongate_vs_frozen_same_query':same})
 for name,val in [('CONCENTRATION_SUMMARY',conc),('GOOD_FAILURE_CASES_PRIVATE',cases),('GATE_SPLIT_SUMMARY',exact)]: (H/(name+'.json')).write_text(json.dumps(val,indent=2)+'\n')
 lines=['# R114 — 原生grounding基础与源域条件RGB桥的分离检验','','沿用三个source100和既有R91，DEV768/384表达、六方向、44保护别名冻结回退。原A256及六层位置不变，FINAL未访问，无专家和新source训练。所有24新臂先封存，再17376端点CPU独立读出与18432任务值几何双实现评分；12旧臂9216任务值复用。','','## 主项和竞争解释','','R113全局撤回point Adapter也改变了RGB桥的key输入及恢复图。本轮让grounding固定原生点—文字基础，RGB桥保留源域point Adapter keys，并用源完整辅助梯度更新45313桥参数；对照为native/sourceRGBkeys×native/source辅助图、同L2错配，以及gateI归零/仅更新gateI/仅更新其余45312参数。所有新提议在C读取前由当前A确定，C只是已有对齐LiDAR、没有新RGB视角；无label接受或逐例选择。固定自然clip1/0.5一步、不回溯/步数网格。全102498参数反向成本保留，不夸称只计算45313梯度。','','## 结果','','| 范围 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---:|---:|---:|']
 for x in aa:
  if x['arm'].endswith('fusedABC') or x['group']=='all_six_equal' and x['arm'] in ['original_source','prepared_frozen','direct_three_geometric']:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 裁决','','主项17.279658%，只比自己的冻结hybrid高0.007960pp，低于R113自然桥17.383951%。源域条件保留没有建立更强TTA，不采用当前修复配方，不加步数/系数网格。nativekeys+sourceaux17.383907几乎复现旧次项，preparedRGBkeys下的nativeaux17.302352也不超过旧次项。辅助图条件变化是实际计算事实，不能据此宣称它是任务退化的主要原因。','','更明确的受控定位：仅gateI更新17.383951%，非gate桥更新17.265658%，分别与旧全桥和零步总体完全相同。完整query相等计数如下；不能只因总体同分就假设所有对象身份相同。固定gateI=0只有16.953391%，直接删除桥有害。下一步应比较输入特定gate更新与统一/打乱gate提议，先区分动态适应和静态校准，不继续扩展注意力更新参数。','','```json',json.dumps(exact,indent=2),'```','','## 正负尾部','','```json',json.dumps(conc,indent=2),'```','','good/bad按全固定分母记录并保留匿名完整逐例标量；本轮没有新看原图，实例身份保持UNKNOWN，不能把IoU改变自动称为目标身份纠错。历史R113图册不作为本轮新主项的语义核验。','','## 工程边界','','12真实输入核验实际分离main和缓存重放位等、RGB源keys冻结late与既有full路径位等、source辅助完整梯度重现及源状态恢复。两个失败attempt来自试图将单独aux梯度与原aux+初始selfkeep梯度逐位比较；极小4.486e-21舍入差仍可改变delta低位。最终使用原physical_direction91完整计算图而非放宽比较，梯度位等通过；原失败代码和日志留存，修正在任务评分前。所有新CPU概率误差<5e-6、无近平赢家歧义。','','主方法及全部对照按原锁定解释，不换主项；当前结论只适用于历史暴露DEV，不是独立确认或稳定TTA成功。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT114_COMPLETE',flush=True)
if __name__=='__main__':main()
