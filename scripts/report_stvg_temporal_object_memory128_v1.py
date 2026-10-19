"""Independent fixed-input/prior/MBR audit and public result report."""
from pathlib import Path
import sys,numpy as np,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_object_memory128_v1 import D,P,DEV,PRE,FIN

def main():
 import torch
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf,probabilities
 from vg_tta.stvg_posterior_decision_tree_v1 import brute_utility
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');res=read(P/'RESULTS_PUBLIC.json');nprior=nspace=nmbr=0;maxerr=0.
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');inp=load(D/'temporal_inputs'/f'{a}.pt');old=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');pr=load(DEV/'priors'/f'{a}.pt');assert z['Source_exact']and z['native_temporal_replay_exact'];assert np.array_equal(z['outputs']['Native']['q'],pr['q'])
  for arm,b in z['trajectories'].items():assert np.array_equal(b,old['E']if arm=='E'else sp['outputs'][f'T20_Live_{arm[4:]}']);nspace+=1
  for arm,vv in z['prior_meta'].items():
   for v,kw,g in zip(vv,inp['inputs'],inp['grids']):
    ww=v['weights'].numpy();allowed=~kw['encoded_mask'][:,-g['h']*g['w']:].numpy();target=allowed.sum(-1);assert np.max(abs((ww*allowed).sum(-1)-target))<1e-3;assert np.isfinite(ww).all()and(ww>0).all();nprior+=1
  ids=z['frame_ids'];ii,jj=np.triu_indices(len(ids),1)
  for arm,o in z['outputs'].items():
   err=float(abs(independent_cdf(probabilities(old['records'],o['logits']),ids)-o['q']).max());assert err<1e-12;maxerr=max(maxerr,err);pick=np.unique(np.linspace(0,len(ii)-1,17).astype(int));u=brute_utility(ids,o['q'],pick);best=brute_utility(ids,o['q'],[int(np.where((np.array(ids)[ii]==o['interval'][0])&(np.array(ids)[jj]+1==o['interval'][1]))[0][0])])[0];assert u.max()<=best+1e-11;nmbr+=len(pick)+1
  assert sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256']and sha(PRE/'teacher0'/f'{a}.pt')==r['teacher_sha256']and sha(DEV/'priors'/f'{a}.pt')==r['prior_sha256']
 assert all(sha(R/f)==h for f,h in p['pins'].items());primary=res['paired']['LiveMean_ReferROI_minus_LiveMean_Native'];control=res['paired']['LiveMean_ReferROI_minus_LiveMean_ShiftedReferROI'];decision='candidate_for_confirmation'if primary['v']['mean']>=2 and primary['v']['CI95'][0]>0 and control['v']['CI95'][0]>0 else'no_meaningful_primary_increment_in_this_configuration'
 audit=dict(status='pass',spatial_arrays_exact=nspace,prior_mass_checks=nprior,brute_MBR_checks=nmbr,max_independent_CDF_error=maxerr,Source_and_original_temporal_logits_exact=True,all_source_spatial_teacher_prior_inputs_unchanged=True,network_weights_frozen=True,new_spatial_updates=0,engineering_recovery='native expanded-query zero stride restored; original failure/debug preserved, cost included',primary_decision=decision)
 put(P/'ROOT_AUDIT.json',audit);put(P/'DECISION_PUBLIC.json',dict(decision=decision,primary=primary,correct_vs_shifted=control,not_all_temporal_OPD_claim=True,no_parameter_sweep=True,independent_confirmation=False,spatial_branch_frozen=True))
 lines=['# 对象条件时间memory：HC128能力筛查','','日期2026-10-19，128已曝光开发父视频，每父一query。原冻结空间Live41/42/43，父内seed均值；E辅助。不是时间OPD，没有新专家、网络训练或空间更新。','','旧Object_temporal只重建共享itq。本轮保持itq、第一pass路由、时间self-attention、全部原memory values与时间头，第二pass每层cross-attention在运动token上加逐帧对象几何log prior。文本prior1，运动prior=.5+.5×fractional覆盖密度，空交集noop，原padding保留。Refer轨迹只作为时间条件，未替换最终空间坐标。','','| 时间条件 | 完整vIoU% | tIoU% | 对原生完整v增量pp [95%CI] |','|---|---:|---:|---|']
 for arm in ['Native','E_ROI','LiveOwnROI','ReferROI','ShiftedReferROI']:
  n=f'LiveMean_{arm}';v=res['means'][n];c=res['paired'].get(n+'_minus_LiveMean_Native');delta='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {delta} |')
 lines+=['','主候选预定ReferROI，不从E/Live/Shifted事后选择赢家。正确ReferROI−ShiftedReferROI完整v：'+json.dumps(control['v']), '', '主候选全部尾部：','```json',json.dumps(primary['tails'],indent=2),'```','','## 工程与成本','','首条GT前身份核验失败：缓存克隆将原跨帧expand共享query实体化，值相同但logits最大差1.430511e-6；原native stride恢复后exact。原失败/源码/旧plan/debug保存，未通过放宽容差或改变模型解决；模型/目标/名单/几何公式不变，失败及排查GPU计入总成本。准备阶段还纠正一次加载外推重复乘128的工程公式，均在任何GT前。','```json',json.dumps(res['cost'],indent=2),'```','','## 核验与裁决','',json.dumps(audit,ensure_ascii=False,indent=2),'','主候选决策：'+decision+'。本轮开发，不宣称独立确认或OPD增量；不改变冻结空间/研究/生产登记。若没有足够主候选收益，关闭此固定prior配置，不追加强度/层/温度扫描；自主目标可转到实质不同机制，旧负结果保留。','','全部128父、全部七原始时间arm×四固定空间、各seed和signed/unknown标量在RESULTS_PUBLIC.json；GT只在全输出seal后评价。源特征和原生八点专家缓存重用与新Source前向成本分别披露，专家缓存不等于原部署专家免费。','']
 (P/'REPORT.md').write_text('\n'.join(lines));put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),code_pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),all_outputs_before_GT=True,exposure='allRC128 development; no independent confirmation',private_excluded=['media','GT coordinates','captions','parent IDs','weights','raw motion/attention/trajectory caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(dict(primary=primary['v'],control=control['v'],audit=audit),indent=2))
if __name__=='__main__':main()
