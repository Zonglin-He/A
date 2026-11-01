"""Use immutable R39 identity aggregate evidence; no private annotations serialized."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'research/round39/independent_identity';EXP=ROOT/'research/round40/experiment'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 seal=read(D/'SEALED.json');assert seal['annotation_sha256']==sha(ROOT/'research/round37/independent_identity/IDENTITY_RECORDS_PRIVATE.json')
 for p,h in seal['files'].items():assert sha(D/p)==h,p
 rows=read(D/'BBOX_IDENTITY_SUMMARY.json');fields=['arm','platform','n','clear_identity_n','ambiguous_n','clear_claimed_mask_n','clear_claim_matches','clear_claim_contradicts','clear_no_mask_claim','center_target_mask_hits_including_overlap','correct_2D_claim_mapped3D_n','correct_2D_claim_but_empty3D_or_subset','mapped_wrong_identity_n','all_review_mapped','all_review_query_changes','all_review_mean_gain_vs_source_pp','all_review_improved','all_review_harmed','not_expert_or_new_blind_gold'];assert all(set(r)==set(fields) for r in rows)
 text='''继续复用原预封24条身份注释，18明确、6歧义。它们披露了历史source／task产物暴露，是非专家身份假说，不是新盲测或expert gold。R39解盲join在新预测封存后完成，注释SHA未变，未新看图或按IoU改身份。

| 接口（18明确分母固定） | 与审阅一致唯一mask | 矛盾mask | 无唯一mask | 一致且有完整3D组 |
|---|---:|---:|---:|---:|
| 兼容point | 8 | 8 | 2 | 8 |
| SoM | 6 | 12 | 0 | 6 |
| R39 bbox中心 | 10 | 7 | 1 | 10 |

bbox中心有11次命中审阅目标mask，但1次同时命中另一mask，按预锁规则保留unknown，不能把11都算正确。10个一致的唯一mask全部可进入full组，旧subset仅5/10；完整映射也放行6条矛盾身份，覆盖不等于可靠。

24例全体bbox中心相对source均值＋4.125835pp，4好3坏；Drone−2.307433pp、Quad＋10.559103pp。该子集好于point＋2.218902pp，但全A208 bbox更差，不能用小子集反选或救回总体未通过的门。6条歧义保留，不计对错或事后解开。

中心一致不认证整个bbox语义，中心不一致也不能独立区分主体理解、坐标、遮挡、不可见中心和已有mask污染。上述R39中心身份表不能直接认证R40矩形最大IoU新提名；本报告只在有另行封存的R40同注释join时追加其结果，不根据它选择映射规则。'''
 result={'R39_join_sealed':True,'annotations_unchanged':True,'annotation_sha256':seal['annotation_sha256'],'R39_join_seal_sha256':sha(D/'SEALED.json'),'n':24,'clear':18,'ambiguous':6,'expert_or_new_blind_gold':False,'prior_artifact_exposure_disclosed':True,'R39_aggregate_rows':[{k:r[k] for k in fields} for r in rows],'R40_identity_join_included':False,'report_text':text,'no_caption_points_masks_coordinates_or_originalID_exported':True}
 (EXP/'IDENTITY_PUBLIC_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('IDENTITY_R39_PUBLIC',len(rows))
if __name__=='__main__':main()
