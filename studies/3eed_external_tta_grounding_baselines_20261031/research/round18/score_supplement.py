"""Sealed C32 diagnostic only; never selects or overwrites main baseline rows."""
from pathlib import Path
import sys,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round18'),str(ROOT/'research/round5/audit')]
from common import EXP,read,dump,sha,verify,seal
from independent_core_v2 import transform_gt,candidate_ious
def main():
 lock=read(EXP/'SUPPLEMENT_SCORE_LOCK.json');assert lock['program_sha256']==sha(__file__);pred=ROOT/'crossplatform/round18/supplement32_attempt001';verify(pred);out=EXP/'scoredC32_attempt001';out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'no_current_C_scores_used_for_optimizer_selection':True})
 spec=importlib.util.spec_from_file_location('R18_locked_summary',EXP.parent/'score.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};rows=[]
 for i,r in enumerate(read(EXP.parent/'input/C32_PRIVATE.json')):
  with np.load(pred/'predictions_PRIVATE'/(r['id']+'.npz')) as z:boxes=z['arm_boxes_raw6'].copy();names=z['arm_names'].tolist();assert str(z['sample_id'])==r['id']
  ix=index[r['id']];assert ix['platform']==r['platform'] and ix['scene']==r['scene'];p=ix['meta_path'];assert sha(p)==ix['metadata_container_sha256']
  if p not in containers:containers[p]=read(p)
  target=transform_gt(containers[p]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));raw=boxes.copy();boxes[:,3:]=np.maximum(boxes[:,3:],np.float32(1e-6));v=candidate_ious(boxes,target)[0];rows.append({'alias':f'C32_{r["platform"]}_{i:03d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':'C32_postprimary_official_config_diagnostic','ious':dict(zip(names,map(float,v))),'box_changed':{n:bool(not np.array_equal(b,raw[0])) for n,b in zip(names,raw)}})
 dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',m.summarize(rows));seal(out,'R18_C32_scored_after_supplement_prediction_seal',n=32,not_primary_replacement=True);print('C32_SCORED')
if __name__=='__main__':main()
