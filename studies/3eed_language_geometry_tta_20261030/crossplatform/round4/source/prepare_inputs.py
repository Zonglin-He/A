"""Extract permitted caption only, create three GT-free raw-point sampling views.
Metadata containers include labels; this preparer does NOT index numeric boxes/classes.
"""
from pathlib import Path
import json,hashlib,sys,datetime,re
import numpy as np
R=Path(__file__).resolve().parent;P=R.parents[1];sys.path.insert(0,str(P/'upstream'))
from utils.transform_waymo import transform_to_front_view
LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seed(s,v):return int(hashlib.sha256(('3eed-round4-resampling-v1|'+s+'|'+str(v)).encode()).hexdigest()[:16],16)
def main():
 lock=R/'PATH_LOCK.json';assert sha(lock)==(R/'PATH_LOCK.sha256').read_text().strip();plan=json.loads(lock.read_text());dest=R/'inputs';dest.mkdir(exist_ok=False)
 receipt={'path_lock_sha256':sha(lock),'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'numeric_bbox_or_class_referenced':False,'annotation_containers_opened':True,'cohorts':{},'point_count':16384,'views':3,'coordinates':'Official front view XY rotation only for Waymo, no unused translation vectors; Drone z+=1.8; Quad unchanged. No centering/scaling.','files':{}}
 for part in ['dev','confirm']:
  rows=[];scoring=[]
  for s in plan['cohorts'][part]:
   obj=json.loads(Path(s['meta_path']).read_text())['ground_info'];assert obj and isinstance(obj[0].get('caption'),str) and obj[0]['caption'].strip(),s['id']
   caption=' '.join(obj[0]['caption'].replace(',',' ,').split());text=caption+' . not mentioned'
   hit=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',caption,re.I)
   noun=hit.group() if hit else None;noun_caption=noun if noun else caption # Explicit identity intervention if no lexicon match.
   noun_text=noun_caption+' . not mentioned'
   rawpath=Path(s['lidar_path']);raw=np.load(rawpath,allow_pickle=False) if rawpath.suffix=='.npy' else np.fromfile(rawpath,np.float32).reshape(-1,4)
   assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
   views=[]
   for v in range(3):
    sd=seed(s['id'],v);ix=np.random.default_rng(sd).choice(len(raw),16384,replace=len(raw)<16384);pc=raw[ix,:3].copy()
    if s['platform']=='drone':pc[:,2]+=1.8
    angle=0
    if s['platform']=='waymo':
     vs=int(rawpath.parent.name.split('_')[-1]);view=['F','FL','FR','SL','SR'][vs];pc,_=transform_to_front_view(pc,np.zeros((1,7)),view);angle=[0,-45,45,-90,90][vs]
    fp=dest/(s['id']+f'_v{v}.npy');np.save(fp,pc.astype(np.float32),allow_pickle=False)
    ip=dest/(s['id']+f'_v{v}_indices.npy');np.save(ip,ix.astype(np.int64),allow_pickle=False)
    receipt['files'][fp.name]=sha(fp);receipt['files'][ip.name]=sha(ip)
    views.append({'view':v,'seed':sd,'point_path':str(fp),'point_sha256':sha(fp),'raw_indices_path':str(ip),'raw_indices_sha256':sha(ip),'raw_index_unique_count':len(np.unique(ix))})
   rows.append({'id':s['id'],'platform':s['platform'],'split':s['split'],'scene':s['scene'],'frame':s['frame'],'text':text,'caption_character_length':len(caption),'noun':noun,'noun_text':noun_text,'noun_caption_character_length':len(noun_caption),'noun_intervention_fallback_identity':noun is None,'views':views,'raw_point_count':len(raw),'raw_file_sha256':sha(rawpath),'official_xy_rotation_degrees':angle,'research_scope':'primary_val' if s['platform']=='waymo' else 'train_union_exposed_auxiliary'})
   scoring.append({'id':s['id'],'platform':s['platform'],'split':s['split'],'scene':s['scene'],'meta_path':s['meta_path'],'object_index':0,'waymo_view_suffix':int(rawpath.parent.name.split('_')[-1]) if s['platform']=='waymo' else None})
  mp=dest/(part+'_manifest.json');mp.write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n');sp=R/(part+'_scoring_index_PRIVATE.json');sp.write_text(json.dumps(scoring,indent=2)+'\n');receipt['cohorts'][part]={'n':len(rows),'manifest_sha256':sha(mp),'scoring_index_sha256':sha(sp),'raw_point_counts':{p:[x['raw_point_count'] for x in rows if x['platform']==p] for p in ['waymo','drone','quad']}}
 (R/'INPUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'prepared':True,'counts':{k:v['n'] for k,v in receipt['cohorts'].items()},'n_viewfiles':len(receipt['files']),'numeric_GT_used':False}))
if __name__=='__main__':main()
