import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round22'),str(ROOT/'research/round22')]
from common22 import *
from score22 import summarize
def main():
    lock('SCORE_SCENE_LOCK.json');pred=BASE/'scene_direct21_attempt001';verify(pred);out=EXP/'scored_scene21_attempt001';out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(pred/'SEALED.json'),'GT_selected_mechanism_panel_not_test_estimate':True,'n':21});O={r['id_PRIVATE']:r for r in read(EXP/'OFFLINE_RECOVERABILITY_PRIVATE.json')};rows=[]
    for r in read(EXP.parent/'input/SCENE21_PRIVATE.json'):
        o=O[r['id']];z=feature(r)
        with np.load(pred/'predictions_PRIVATE'/(r['id']+'.npz')) as p:names=p['arm_names'].tolist();q=p['query_indices'];assert np.array_equal(p['arm_boxes_raw6'],z['boxes'][q]);values=np.asarray(o['candidate_ious_PRIVATE'])[q]
        rows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'phase':r['stage'] if r['stage'] in ['W','Z'] else 'D','scene_PRIVATE':r['scene'],'ious':dict(zip(names,map(float,values))),'q':dict(zip(names,map(int,q))),'offered_recoverable25':o['offered_recoverable25'],'all256_recoverable25':o['all256_recoverable25'],'has_reference_relation':o['has_reference_relation']})
    s=summarize(rows,'scene_no_ref');dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',s);seal(out,'R22_label_enriched_scene_mechanism_scored_not_confirmatory',n=21);print('SCENE_SCORE22',{p:{a:v['mIoU_percent'] for a,v in x.items()} for p,x in s['ALL'].items()})
if __name__=='__main__':main()
