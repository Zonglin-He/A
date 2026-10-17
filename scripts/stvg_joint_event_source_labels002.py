"""Bind source labels by actual clip identity, not duplicate caption lookup.

Only downstream source labels change. All source/target model computations,
critic architecture/budget and the original runtime/deadline remain unchanged.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import BASE,read,put,sha,load,evaluate_payload,verify

def bind():
 p=verify();ap=ROOT/'external/VidSTG-Dataset/annotations/train_annotations.json';ann=read(ap);rows=[];disagreements=[]
 import ijson
 oldkeys={r['annotation_key']for r in p['source']if r['dataset']=='vidstg'}
 with open(p['source_provenance']['vidstg']['annotation'],'rb')as f:
  old={k:v for k,v in ijson.kvitems(f,'',use_float=True)if k in oldkeys}
 for r in p['source']:
  if r['dataset']!='vidstg':continue
  q=r['input'];matches=[]
  for i,v in enumerate(ann):
   if v['vid']!=r['parent']:continue
   if int(v['used_segment']['begin_fid'])!=q['start_frame']or int(v['used_segment']['end_fid'])!=q['end_frame']:continue
   for field in ['captions','questions']:
    for j,c in enumerate(v.get(field,[])):
     if c['description'].strip().lower()==q['caption'].strip().lower():matches.append((i,field,j,int(c['target_id'])))
  assert matches,('No exact original source clip/query binding',r['alias'])
  identities={(ann[i]['temporal_gt']['begin_fid'],ann[i]['temporal_gt']['end_fid'],target)for i,field,j,target in matches}
  assert len(identities)==1,('Ambiguous source target, never resolve with GT performance',r['alias'])
  i,field,j,target=matches[0];v=ann[i]
  oldv=old[r['annotation_key']];o=oldv['ori_temp_gt'];correct=v['temporal_gt'];different=(int(o['begin_fid']),int(o['end_fid']))!=(int(correct['begin_fid']),int(correct['end_fid']))
  if different:disagreements.append(r['alias'])
  rows.append(dict(alias=r['alias'],original_annotation_index=i,field=field,query_index=j,target_id=target,duplicate_equivalent_bindings=len(matches),video_query_segment_exact=True,old_processed_pointer_wrong_event=different))
 put(BASE/'SOURCE_LABEL_BINDING002.json',dict(status='pass',count=48,rows=rows,raw_annotations=str(ap.relative_to(ROOT)),raw_annotations_sha256=sha(ap),VidOR_annotations='downloads/vidor/training-annotation.zip',VidOR_sha256=sha(ROOT/'downloads/vidor/training-annotation.zip'),old_wrong_event_count=len(disagreements),old_wrong_event_aliases=disagreements,selection_uses_no_prediction_or_GT_quality=True,training_started=False,source_inputs_and_target_outputs_unchanged=True,original_plan_sha256=sha(BASE/'PLAN_LOCK.json'),original_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json')))
 print('SOURCE_LABEL_BINDING_PASS',48,'wrong old event pointers',len(disagreements),flush=True)

def labels():
 import zipfile
 p=verify();binding=read(BASE/'SOURCE_LABEL_BINDING002.json');assert binding['status']=='pass'
 assert sha(ROOT/binding['raw_annotations'])==binding['raw_annotations_sha256']
 assert sha(ROOT/binding['VidOR_annotations'])==binding['VidOR_sha256']
 bar=read(BASE/'SOURCE_PREDICTION_BARRIER.json');assert bar['count']==96
 zs={z['alias']:z for z in [load(ROOT/r['path'])for r in bar['files']]};a=read(ROOT/binding['raw_annotations']);br={r['alias']:r for r in binding['rows']};labels=[]
 from scripts.prepare_vidstg_wrong_domain_support import trajectory_for_target
 archive=zipfile.ZipFile(ROOT/binding['VidOR_annotations']);names={Path(n).stem:n for n in archive.namelist()if n.endswith('.json')};hc=read(p['source_provenance']['hc2']['annotation']);raw_cache={}
 for r in p['source']:
  ds=r['dataset'];z=zs[r['alias']]
  if ds=='hc2':
   v=hc[r['annotation_key']];assert v['English'].lower()==r['input']['caption'].lower();s=int(v['st_frame'])-1;e=s+len(v['bbox'])-1
   dense={s+i:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])]for i,(x,y,w,h)in enumerate(v['bbox'])}
  else:
   b=br[r['alias']];v=a[b['original_annotation_index']];c=v[b['field']][b['query_index']];assert v['vid']==r['parent']and c['description'].strip().lower()==r['input']['caption'].strip().lower()and int(c['target_id'])==b['target_id']
   raw=json.loads(archive.read(names[r['parent']]))
   assert raw['width']==r['input']['width']and raw['height']==r['input']['height']
   s,e=int(v['temporal_gt']['begin_fid']),int(v['temporal_gt']['end_fid']);dense={}
   for fid,t in trajectory_for_target(raw,b['target_id']).items():
    if s<=int(fid)<e:
     x,y,w,h=t['bbox'];dense[int(fid)]=[x,y,x+w,y+h]
   assert dense,('No source GT trajectory; preserve, do not manufacture negatives',r['alias'])
  assert e>s
  labels.append(dict(alias=r['alias'],dataset=ds,split=r['split'],values=evaluate_payload(z,dense,[s,e]),source_GT=True,official_original_frame_binding=True))
 put(BASE/'SOURCE_LABELS.json',dict(status='complete',count=96,rows=labels,source_GT_only=True,target_GT_training=False,prediction_barrier_sha256=sha(BASE/'SOURCE_PREDICTION_BARRIER.json'),binding002_sha256=sha(BASE/'SOURCE_LABEL_BINDING002.json'),endpoint_conventions='same as original audited HC2 and VidSTG scorers, no relative/global coordinate mixing'))
 print('SOURCE_LABELS_COMPLETE',len(labels),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
