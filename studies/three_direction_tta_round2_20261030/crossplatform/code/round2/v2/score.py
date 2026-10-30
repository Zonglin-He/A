"""Separate-process CPU scoring, only after every cohort prediction is sealed."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

R=Path(__file__).resolve().parent;P=R.parent.parent
sys.path[:0]=[str(P/'upstream'),str(P.parent/'tools')]
from utils.transform_waymo import transform_to_front_view
from obb_iou import oriented_iou

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rotation(yaw):
    yaw=float(yaw)  # np.float32 trig violates strict OBB halfspace tolerance.
    c,s=np.cos(yaw),np.sin(yaw);matrix=np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)
    assert np.max(abs(matrix@matrix.T-np.eye(3)))<1e-12
    return matrix
def metric(box,gt):
    if not np.isfinite(box).all() or np.any(box[3:6]<=0):return 0.,True
    return float(oriented_iou(box[:3],box[3:6],rotation(box[6]),gt[:3],gt[3:6],rotation(gt[6]))),False
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args()
    out=R/(a.cohort+'_predictions');seal=json.loads((out/'sealed.json').read_text());plan=json.loads((R/'PLAN_LOCK.json').read_text());assert seal['plan_sha256']==sha(R/'PLAN_LOCK.json');assert seal['arms']==plan['arms']
    lock=json.loads((R/'SCORER_LOCK.json').read_text());assert sha(Path(__file__))==lock['scorer_sha256'];assert sha(P.parent/'tools/obb_iou.py')==lock['obb_implementation_sha256'];assert sha(R/'PLAN_LOCK.json')==lock['plan_sha256']
    index=R.parent/(a.cohort+'_scoring_index_PRIVATE.json');expected=json.loads((R.parent/'INPUT_PREPARATION_RECEIPT.json').read_text())['cohorts'][a.cohort]['scoring_index_sha256'];assert sha(index)==expected;samples=json.loads(index.read_text());assert len(samples)==seal['n']
    for s in samples:assert sha(out/(s['id']+'.npz'))==seal['files'][s['id']+'.npz']
    assert len(seal['files'])==len(samples)
    # First numericbox access in this program occurs after all seal checks above.
    rows=[]
    for s in samples:
        meta=json.loads(Path(s['meta_path']).read_text());gt=np.asarray(meta['ground_info'][s['object_index']]['bbox_3d'][:7],dtype=float)
        if s['platform']=='drone':gt[2]+=1.8
        if s['platform']=='waymo':
            _,transformed=transform_to_front_view(np.zeros((1,3)),gt[None],['F','FL','FR','SL','SR'][s['Waymo_view_suffix']]);gt=transformed[0]
        assert np.isfinite(gt).all() and np.all(gt[3:6]>0)
        pred=np.load(out/(s['id']+'.npz'),allow_pickle=False)
        for arm in seal['arms']:
            val,invalid=metric(pred[arm],gt);rows.append({'id':s['id'],'platform':s['platform'],'scene':s['scene'],'arm':arm,'iou':val,'acc25':val>.25,'acc50':val>.5,'invalid_geometry':invalid})
    summaries=[];paired=[]
    for platform in ['all','waymo','drone','quad']:
        subset=[x for x in rows if platform=='all' or x['platform']==platform];base={x['id']:x for x in subset if x['arm']=='source'}
        for arm in seal['arms']:
            ar=[x for x in subset if x['arm']==arm];d=np.array([x['iou']-base[x['id']]['iou'] for x in ar]);summaries.append({'platform':platform,'arm':arm,'n':len(ar),'mean_iou':float(np.mean([x['iou'] for x in ar])),'acc25_count':sum(x['acc25'] for x in ar),'acc50_count':sum(x['acc50'] for x in ar),'invalid_count':sum(x['invalid_geometry'] for x in ar)})
            scenes=sorted(set(x['scene'] for x in ar));groups={sc:[x for x in ar if x['scene']==sc] for sc in scenes};rng=np.random.default_rng(20261030);boot=[]
            for _ in range(1000):
                chosen=rng.choice(scenes,len(scenes),replace=True);vals=[x['iou']-base[x['id']]['iou'] for sc in chosen for x in groups[sc]];boot.append(np.mean(vals))
            tails=sorted([{'id':x['id'],'source_iou':base[x['id']]['iou'],'arm_iou':x['iou'],'delta_iou':x['iou']-base[x['id']]['iou']} for x in ar],key=lambda x:x['delta_iou'])
            paired.append({'platform':platform,'arm':arm,'mean_delta':float(d.mean()),'improved':int((d>1e-10).sum()),'harmed':int((d < -1e-10).sum()),'equal':int((abs(d)<=1e-10).sum()),'rescue25':sum(x['acc25'] and not base[x['id']]['acc25'] for x in ar),'harm25':sum(not x['acc25'] and base[x['id']]['acc25'] for x in ar),'rescue50':sum(x['acc50'] and not base[x['id']]['acc50'] for x in ar),'harm50':sum(not x['acc50'] and base[x['id']]['acc50'] for x in ar),'scene_clusters':len(scenes),'scene_cluster_bootstrap_95ci_descriptive':np.percentile(boot,[2.5,97.5]).tolist(),'tails':tails})
    trace=json.loads((out/'trace.json').read_text());acceptance={}
    for platform in ['all','waymo','drone','quad']:
        tr=[x for x in trace if platform=='all' or x['platform']==platform];acceptance[platform]={}
        for teacher in ['old','roof']:
            acceptance[platform][teacher]={'frames':len(tr),'frames_with_accepted_geometry':sum(any(x['teachers'][teacher]['accepted']) for x in tr),'accepted_candidates':sum(sum(x['teachers'][teacher]['accepted']) for x in tr),'direct_identity_switches':sum(x['teachers'][teacher]['direct_query']!=x['source_query'] for x in tr)}
        acceptance[platform]['roof_added_candidates']=sum(sum(x['teachers']['roof']['accepted'])-sum(x['teachers']['old']['accepted']) for x in tr)
    payload={'cohort':a.cohort,'scope':'official7D target-only sourceweights+textonly readout, strictIoUthresholds; scenecluster intervals descriptive, not allframesindependent','prediction_seal_sha256':sha(out/'sealed.json'),'scorer_lock_sha256':sha(R/'SCORER_LOCK.json'),'n':len(samples),'rows':rows,'summaries':summaries,'paired_vs_source':paired,'acceptance':acceptance}
    dest=R/(a.cohort+'_scored_double.json');assert not dest.exists();dest.write_text(json.dumps(payload,indent=2));print(json.dumps({'cohort':a.cohort,'summaries':[x for x in summaries if x['platform']=='all'],'acceptance':acceptance},indent=2))
if __name__=='__main__':main()
