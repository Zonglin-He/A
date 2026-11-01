"""GT accessed only after complete direct predictions have been sealed."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round21'),str(ROOT/'research/round5/audit')]
from common21 import *
from independent_core_v2 import transform_gt,candidate_ious
def summarize(rows):
    out={}
    for phase in ('ALL','D','W','Z'):
        out[phase]={}
        for platform in ('drone','quad'):
            rr=[r for r in rows if r['platform']==platform and (phase=='ALL' or r['phase']==phase)];scenes=sorted({r['scene_PRIVATE'] for r in rr});arms={}
            for arm in rr[0]['ious']:
                vals=np.array([r['ious'][arm] for r in rr]);src=np.array([r['ious']['source'] for r in rr]);direct=np.array([r['ious']['gate_repair_no_reference'] for r in rr]);delta=vals-direct;clusters=[delta[[r['scene_PRIVATE']==scene for r in rr]] for scene in scenes];rng=np.random.default_rng(2021);boot=[np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))]).mean()*100 for _ in range(10000)]
                arms[arm]={'n':len(rr),'mIoU_percent':float(vals.mean()*100),'Acc25_percent':float((vals>.25).mean()*100),'Acc50_percent':float((vals>.5).mean()*100),'gain_vs_source_pp':float((vals-src).mean()*100),'gain_vs_no_reference_pp':float(delta.mean()*100),'gains_vs_no_reference':int((delta>1e-10).sum()),'harms_vs_no_reference':int((delta< -1e-10).sum()),'changed_vs_no_reference':sum(r['q'][arm]!=r['q']['gate_repair_no_reference'] for r in rr),'cluster95_relation_gain_pp':list(map(float,np.quantile(boot,[.025,.975]))),'scenes':len(scenes)}
            out[phase][platform]=arms
    return out
def main():
    lock('SCORE_v2_LOCK.json');pred=BASE/'direct208_attempt002';verify(pred);out=EXP/'scored208_attempt002';out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(pred/'SEALED.json'),'score_program_sha256':sha(__file__),'all208_historically_exposed':True});idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};rows=[];oldscore={x['id_PRIVATE']:x for x in read(ROOT/'research/round20/experiment/scored208_attempt002/ROWS_PRIVATE.json')}
    for r in read(ROWS):
        with np.load(pred/'predictions_PRIVATE'/(r['id']+'.npz')) as z:names=z['arm_names'].tolist();boxes=z['arm_boxes_raw6'].copy();q=z['query_indices'].tolist()
        ix=idx[r['id']];p=ix['meta_path'];assert sha(p)==ix['metadata_container_sha256']
        if p not in containers:containers[p]=read(p)
        gt=transform_gt(containers[p]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes[:,3:]=np.maximum(boxes[:,3:],np.float32(1e-6));v=candidate_ious(boxes,gt)[0];ious=dict(zip(names,map(float,v)));assert abs(ious['source']-oldscore[r['id']]['ious']['source'])<1e-12;assert abs(ious['r20_fullG']-oldscore[r['id']]['ious']['direct_fullG'])<1e-12
        rows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'phase':r['stage'] if r['stage'] in ('W','Z') else 'D','scene_PRIVATE':r['scene'],'ious':ious,'q':dict(zip(names,q))})
    summary=summarize(rows);gate={}
    for platform in ('drone','quad'):
        s=summary['ALL'][platform]['direct_full_relation_G'];checks={'positive_relation_increment':s['gain_vs_no_reference_pp']>0,'no_harmful_relation_decisions':s['harms_vs_no_reference']==0,'positive_cluster_lower':s['cluster95_relation_gain_pp'][0]>0};gate[platform]={'development_passed':all(checks.values()),'checks':checks,'independent_confirmation_completed':False,'TTA_allowed':False}
    dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',summary);dump(out/'TTA_GATE.json',gate);seal(out,'R21_sealed_scored_all208_all_10arms_negatives_retained',n=208);print('SCORE21',summary['ALL'],gate)
if __name__=='__main__':main()
