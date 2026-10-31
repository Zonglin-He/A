"""Register fixed existing A32/B32; no task label/media/model reads."""
from pathlib import Path
import json,hashlib,sys,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round16'))
from common import dump,read,sha,EXP
def main():
 d=EXP.parent/'input';d.mkdir(parents=True,exist_ok=False)
 old=ROOT/'research/round15/input'
 A=read(old/'CONDITIONAL_A32_GPU_INPUTS_PRIVATE.json');B=read(old/'UNUSED_B32_GPU_INPUTS_PRIVATE.json')
 cases={x['id']:x for x in read(old/'FORMAL_CASE204_PRIVATE.json')};index=read(old/'G_FRAME_INDEX_PRIVATE.json')
 assert len(A)==len(B)==32 and not {x['id'] for x in A}&{x['id'] for x in B}
 assert collections.Counter(x['platform'] for x in A)==collections.Counter(x['platform'] for x in B)=={'drone':16,'quad':16}
 dump(d/'A32_PRIVATE.json',A);dump(d/'B32_PRIVATE.json',B);dump(d/'ALL64_PRIVATE.json',A+B)
 ac=[dict(cases[x['id']],case_alias=f"A_{x['platform']}_{i:03d}") for i,x in enumerate(A)]
 dump(d/'A_CASES_PRIVATE.json',ac);dump(d/'A_FRAME_INDEX_PRIVATE.json',{x['frame_alias']:index[x['frame_alias']] for x in ac})
 dump(d/'SPLIT_DISCLOSURE.json',{'A':32,'B':32,'per_platform_each':16,'A_G_and_fit_only':True,'B_no_RGB_G_or_fitting':True,'same_instance_correspondence_claim':False,'cross_observation_domain_shared_update':True,'scenes_per_platform':{'drone':1,'quad':2},'B_previous_R15_source_only_no_R15_G_fit_score':True,'all_5912_historically_task_scored':True,'not_pristine_blind_test':True,'A_previous_G_and_recent_scoring_exposed':True,'selection':'Exact R15 metadata-locked support/reserved inputs; no new selection by caption or score'})
 dump(EXP/'AUTHORIZATION.json',{'user_instruction':'那就按这个做，先冻结框，只更新匹配投影层，在Waymo→Drone和Quad上试。先验证G能产生可靠的候选偏好，保留同信息直接重排、不更新参数的对照，用没参与监督构造和拟合的新观测验证，所有负例都保留。','source_immutable':True,'no_teacher_EMA':True,'no_source_retraining':True})
 print('registered A32/B32')
if __name__=='__main__':main()
