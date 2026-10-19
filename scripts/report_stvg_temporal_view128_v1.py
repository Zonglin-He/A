"""Independent physical-support/mixture audit and whole-cohort view report."""
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_view128_v1 import D,P,DEV,PRE,FIN

def main():
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf,probabilities
 from vg_tta.stvg_posterior_decision_tree_v1 import brute_utility
 p=read(D/'PLAN_LOCK.json');res=read(P/'RESULTS_PUBLIC.json');checks=dict(Source_posteriors=0,physical_mass_mixtures=0,common_decision_grid=0,direct_utilities=0,spatial_arrays_exact=0,grid_only_exact=0);maxerr=0.;no_call=0;new_frame_counts=[]
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ctrl=load(D/'no_RGB_controls'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');assert z['grid_only_exact']and z['outputs']['Native']['interval']==old['interval'];checks['grid_only_exact']+=1
  for name,b in z['trajectories'].items():assert np.array_equal(b,inp['E']if name=='E'else sp['outputs'][f'T20_Live_{name[4:]}']);checks['spatial_arrays_exact']+=1
  for name,v in z['observations'].items():
   err=float(abs(independent_cdf(probabilities(v['records'],v['logits']),v['ids'])-v['q']).max());assert err<1e-12;maxerr=max(maxerr,err);checks['Source_posteriors']+=1;assert v['ids']==r['observation_grids'][name];no_call+=int(not v['new_Source'])
  for name,o in z['outputs'].items():
   U=np.array(o['hypothesis_ids']);N=len(U);expect=np.zeros((N,N));parts=[(1.,old['ids'],old['q'])]if name=='Native'else[(1.,z['observations'][name[:-6]]['ids'],z['observations'][name[:-6]]['q'])]if name.endswith('Direct')else[(.5,old['ids'],old['q']),(.5,z['observations'][name[:-3]]['ids'],z['observations'][name[:-3]]['q'])]
   for weight,grid,q in parts:
    index=np.searchsorted(U,grid);expect[np.ix_(index,index)]+=weight*q
   assert np.max(abs(expect-o['q']))<1e-15;checks['physical_mass_mixtures']+=1;assert o['original_decision_grid']==old['ids'];checks['common_decision_grid']+=1;ii,jj=np.triu_indices(len(old['ids']),1);picks=np.unique(np.linspace(0,len(ii)-1,17).astype(int));old_ids=np.array(old['ids']);hu,hv=np.triu_indices(N,1);hyp=(U[hu],U[hv]+1);prob=o['q'][hu,hv]
   for k in picks:
    aa,bb=old_ids[ii[k]],old_ids[jj[k]]+1;inter=np.maximum(0,np.minimum(bb,hyp[1])-np.maximum(aa,hyp[0]));union=bb-aa+hyp[1]-hyp[0]-inter;value=float(np.dot(prob,inter/union));err=abs(value-o['decision_utility'][ii[k],jj[k]]);assert err<1e-11;maxerr=max(maxerr,err);checks['direct_utilities']+=1
  assert sha(DEV/'priors'/f'{a}.pt')==r['prior_sha256']and sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256']and sha(R/r['subject_path'])==r['subject_sha256'];new_frame_counts.append(r['view_meta']['new_Window_RGB_frames'])
 assert all(sha(R/f)==h for f,h in p['pins'].items());audit=dict(status='pass',checks=checks,max_independent_error=maxerr,exact_duplicate_observations=no_call,mean_new_context_RGB_frames=float(np.mean(new_frame_counts)),original_spatial_and_native_inputs_unchanged=True,all128_sealed_before_GT=True,all128_all5arms_retained=True,new_experts=0,new_network_training=0)
 audit['all128_no_RGB_controls_sealed_before_GT']=read(D/'CONTROL_BARRIER.json')['all_before_new_GT'];put(P/'ROOT_AUDIT.json',audit)
 candidates=[a for a in p['arms']if a!='Native'];scores={a:res['means'][f'LiveMean_{a}']['v']for a in candidates};best=max(scores.values());selected=min([a for a,x in scores.items()if abs(x-best)<=1e-12],key=lambda a:(not a.endswith('Mix'),a));gain=res['paired'][f'LiveMean_{selected}_minus_LiveMean_Native']['v'];decision='freeze_for_parent_disjoint_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 else'no_candidate_has_supported_meaningful_development_increment'
 put(P/'SELECTION_PUBLIC.json',dict(selected_development_candidate=selected,selection='highest fullv parent seed mean across4 prespecified views/readouts; ties preferMix then lexical; development selection not independent proof',selected_delta=gain,decision=decision,confirmation_not_started=True,not_OPD=True))
 lines=['# 实际时间观察：HC128冻结空间能力筛查','','同一TA-STVG Source权重，全部RC128已曝光开发父，每父一query。原空间Live41/42/43先父内均值，E辅助；任何输出不改原空间坐标。比较同覆盖较粗33实际帧和围绕原MBR扩2x上下文的最多65实际帧重读；不是缓存特征裁剪，不增加DINO/Refer或训练。','','| 时间读出 | vIoU% | tIoU% | v增量pp [95%CI] |','|---|---:|---:|---|']
 for arm in p['arms']+['CoarseNoRGBDirect','CoarseNoRGBMix','WindowNoRGBDirect','WindowNoRGBMix']:
  key=f'LiveMean_{arm}';v=res['means'][key];c=res['paired'].get(key+'_minus_LiveMean_Native');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {d} |')
 lines+=['','Direct用新观察后验；Mix固定.5原分布+.5新观察后验。NoRGB对照仅将旧各offset端点标准化logits按物理时间插值到新采样点，重建同支持后验，不重编码RGB；用于判断是否仅限制候选/重采样logits，并非真实新视觉证据或选型候选。该对照在前两预测/后续运行中、任何新GT之前增补，原五arm预测不改。局部后验不是事件存在概率，全局保留不承诺无损。所有组最终决策严格原65物理端点2080区间；仅hypothesis物理支持union用于精确expectedIoU，未增加可输出决策。Native及grid-only嵌入精确恢复旧MBR，空间s按构造完全一致。','','## 开发选择与边界','',json.dumps(read(P/'SELECTION_PUBLIC.json'),ensure_ascii=False,indent=2),'','四个预登记候选的pointwise区间不修正开发选择偏差；选中后须父不重叠冻结确认。没有按视频/错误类别挑赢家或删样本，不把观察/读出增益称OPD。','','## 全部尾部','']
 for arm in candidates:lines+=['### '+arm,'','```json',json.dumps(res['paired'][f'LiveMean_{arm}_minus_LiveMean_Native']['tails'],indent=2),'```','']
 lines+=['## 工程、计费与核验','','原900s自定预算在首两无GT计时max外推995.8s时按规则停止，2完整预测与成本保留；仅修订有限总预算1200s/单进程900s续跑，名单/观察规则/公式/参数/Source调用上限不变，不读取GT后救参数。准备阶段合成Direct测试发现union漏纳原决策端点，在任何模型/GT前修正为协议要求，旧码/plan保留preflight001。','```json',json.dumps(res['cost'],ensure_ascii=False,indent=2),'```','','独立核验：','```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','','历史密采样HC负结果与对象memory负结果保留。本轮判断仅限这两类真实时间观察和固定读出，不泛化所有视频条件OPD。当前研究/生产登记及空间配置不改；全部逐父×3seed×5时间/4固定空间signed/unknown在RESULTS_PUBLIC.json，RGB/GT坐标/文本/身份/权重/raw caches不公开。','']
 (P/'REPORT.md').write_text('\n'.join(lines));put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),code_pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),new_GT_after_all128=True,exposure='all RC128 development; past method/source exposure',private_excluded=['media','GT coordinates','captions','parent IDs','weights','raw Source/trajectory/posterior caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(dict(audit=audit,selection=read(P/'SELECTION_PUBLIC.json')),indent=2))
if __name__=='__main__':main()
