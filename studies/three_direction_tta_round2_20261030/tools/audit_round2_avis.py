"""Post-seal independent exact-set metric using augmenting-path matching."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from pycocotools import mask as maskapi

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def nonempty(seg,h,w):
    if seg is None or seg==[]:return None
    if isinstance(seg,dict) and isinstance(seg['counts'],(str,bytes)):r=seg
    else:
        r=maskapi.frPyObjects(seg,h,w)
        if isinstance(r,list):r=maskapi.merge(r)
    assert list(r['size'])==[h,w]
    return r if maskapi.area(r)>0 else None
def perfect(edges):
    n=len(edges);assigned=[-1]*n
    def visit(i,seen):
        for j in range(n):
            if not edges[i,j] or seen[j]:continue
            seen[j]=True
            if assigned[j]<0 or visit(assigned[j],seen):assigned[j]=i;return True
        return False
    return all(visit(i,[False]*n) for i in range(n))
def metric(gt,pr,v):
    levels=np.arange(.05,1.,.05);correct=np.zeros(19,dtype=np.int64);strata={k:{'n':0,'correct':np.zeros(19,dtype=np.int64)} for k in ['none','single','multi']}
    h,w=v['height'],v['width']
    for local,frame in enumerate(v['frame_indices']):
        a=[(t['category_id'],nonempty(t['segmentations'][frame],h,w)) for t in gt]
        b=[(t['category_id'],nonempty(t['segmentations'][local],h,w)) for t in pr]
        a=[x for x in a if x[1] is not None];b=[x for x in b if x[1] is not None]
        kind='none' if not a else 'single' if len(a)==1 else 'multi';strata[kind]['n']+=1
        passed=np.zeros(19,dtype=np.int64)
        if Counter(x[0] for x in a)==Counter(x[0] for x in b):
            if not a:passed[:]=1
            else:
                overlaps=maskapi.iou([x[1] for x in a],[x[1] for x in b],[0]*len(b));classes=np.equal.outer([x[0] for x in a],[x[0] for x in b])
                for k,level in enumerate(levels):passed[k]=perfect(classes&(overlaps>level))
        correct+=passed;strata[kind]['correct']+=passed
    return {'frames':len(v['frame_indices']),'correct':correct.tolist(),'score':float(correct.mean()/len(v['frame_indices'])),'strata':{k:{'n':x['n'],'correct':x['correct'].tolist()} for k,x in strata.items()}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());assert protocol['state']=='sealed'
    plan=Path(protocol['args']['plan']);assert sha(plan)==protocol['plan_sha256'];p=json.loads(plan.read_text())
    manifest=Path(p['manifest']);assert sha(manifest)==p['manifest_sha256']==protocol['manifest_sha256'];inputs=json.loads(manifest.read_text())
    arms=p['arms'];preds={}
    for arm in arms:
        d=a.run/arm;s=json.loads((d/'seal.json').read_text());assert s['state']=='sealed' and s['plan_sha256']==sha(plan) and s['manifest_sha256']==sha(manifest)
        assert s['prediction_sha256']==sha(d/'predictions.json');preds[arm]=json.loads((d/'predictions.json').read_text())
    # Targets are opened only after every arm seal has passed.
    gtpath=ROOT/'avis/datasets/test.json';gt=json.loads(gtpath.read_text());videos={Path(v['file_names'][0]).parts[0]:v for v in gt['videos']}
    rows=[]
    for v in inputs['videos']:
        original=videos[v['video_id']];assert (v['height'],v['width'],v['original_length'])==(original['height'],original['width'],original['length'])
        targets=[t for t in gt['annotations'] if t['video_id']==original['id']]
        for arm in arms:
            pr=[t for t in preds[arm] if t['video_id']==v['video_id']]
            assert all(len(t['segmentations'])==v['length'] for t in pr)
            rows.append({'video_key':hashlib.sha256(v['video_id'].encode()).hexdigest()[:16],'arm':arm,**metric(targets,pr,v)})
    summaries=[]
    for arm in arms:
        rr=[x for x in rows if x['arm']==arm];correct=np.sum([x['correct'] for x in rr],0);frames=sum(x['frames'] for x in rr)
        summaries.append({'arm':arm,'n_videos':len(rr),'frames':frames,'correct_total_over19thresholds':int(correct.sum()),'frame_weighted_score':float(correct.mean()/frames),'video_weighted_score':float(np.mean([x['score'] for x in rr]))})
    result={'all_arm_seals_verified_before_targets':True,'target_sha256':sha(gtpath),'metric':'nonempty exact category-multiplicity and per-threshold feasible matching, IoU>.05:.05:.95; not native FSLA','matching_implementation':'Independent augmenting-path maximum bipartite matching; no Hungarian solver','rows':rows,'summaries':summaries}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
