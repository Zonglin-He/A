"""Extract allowed inputs from mixed JSON containers, without reading numeric box fields.

This is preparation, not a claim that no annotation-containing file was opened.
The adaptation process later reads only the standalone input cache.
"""
from pathlib import Path
import hashlib, json, sys
from datetime import datetime, timezone
import numpy as np

R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P/'upstream'))
from utils.transform_waymo import transform_to_front_view

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=R/'COHORT_PATH_LOCK.json';assert sha(plan)==(R/'COHORT_PATH_LOCK.sha256').read_text().strip()
    cohorts=json.loads(plan.read_text())['cohorts'];dest=R/'inputs';dest.mkdir(exist_ok=False)
    receipt={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'cohort_path_lock_sha256':sha(plan),'access_boundary':'Official JSON containers containing annotations opened only by preparer. Only ground_info[0].caption extracted; numericbbox and class never referenced. Adaptation manifest contains no metadata/GT path.','point_count':16384,'sampling':'numpy.default_rng; withoutreplacement if possible, repeat otherwise','coords':'Dronez+1.8; Waymo suffix rotations matching released loader; no added normalization','files':{},'cohorts':{},'state':'prepared_no_model_or_scoring'}
    for part in ['dev','confirm']:
        output=[];scoring=[]
        for i,s in enumerate(cohorts[part]):
            meta=json.loads(Path(s['meta_path']).read_text());objects=meta['ground_info']
            assert objects and isinstance(objects[0].get('caption'),str) and objects[0]['caption'].strip(),s['id']
            text=' '.join(objects[0]['caption'].replace(',',' ,').split())+' . not mentioned'
            lp=Path(s['lidar_path']);raw=np.load(lp,allow_pickle=False) if lp.suffix=='.npy' else np.fromfile(lp,np.float32).reshape(-1,4)
            assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
            seed=20261030+i+(0 if part=='dev' else 10000)
            rng=np.random.default_rng(seed);indices=rng.choice(len(raw),16384,replace=len(raw)<16384);pc=raw[indices,:3].copy()
            if s['platform']=='drone':pc[:,2]+=1.8
            if s['platform']=='waymo':
                view=['F','FL','FR','SL','SR'][int(lp.parent.name.split('_')[-1])]
                pc,_=transform_to_front_view(pc,np.zeros((1,7)),view)
            pc=pc.astype(np.float32);f=dest/(s['id']+'.npy');np.save(f,pc,allow_pickle=False);receipt['files'][f.name]=sha(f)
            output.append({'id':s['id'],'platform':s['platform'],'scene':s['scene'],'frame':s['frame'],'text':text,'point_path':str(f),'point_sha256':sha(f),'seed':seed})
            scoring.append({'id':s['id'],'platform':s['platform'],'scene':s['scene'],'meta_path':s['meta_path'],'object_index':0,'Waymo_view_suffix':int(lp.parent.name.split('_')[-1]) if s['platform']=='waymo' else None})
        mp=dest/(part+'_manifest.json');mp.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');receipt['cohorts'][part]={'n':len(output),'input_manifest_sha256':sha(mp),'numericlabels_read_or_scored':False}
        sp=R/(part+'_scoring_index_PRIVATE.json');sp.write_text(json.dumps(scoring,indent=2)+'\n');receipt['cohorts'][part]['scoring_index_sha256']=sha(sp)
    (R/'INPUT_PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'state':receipt['state'],'dev':receipt['cohorts']['dev']['n'],'confirm':receipt['cohorts']['confirm']['n'],'cache_files':len(receipt['files']),'gtfields_used':False}))
if __name__=='__main__':main()
