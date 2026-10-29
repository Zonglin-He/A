"""Unlabeled sensitivity audit against matched no-update views; never task accuracy."""
import pathlib,json,hashlib,numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mu
r=pathlib.Path(__file__).resolve().parents[1];study=r/'studies/avis_prefix_dev_v3';root=study/'runs'
def load(name):
 d=root/name;p=d/'predictions.json';proto=json.loads((d/'protocol.json').read_text());assert proto['state']=='sealed' and hashlib.sha256(p.read_bytes()).hexdigest()==proto['prediction_sha256'];return json.loads(p.read_text())
base=load('views_0');videos=[v['video_id'] for v in json.loads((study/'inputs.json').read_text())['videos']];out=[]
def tube_iou(a,b):
 inter=union=0
 for x,y in zip(a['segmentations'],b['segmentations']):
  aa=int(mu.area(x));bb=int(mu.area(y));ii=int(mu.area(mu.merge([x,y],intersect=True)));inter+=ii;union+=aa+bb-ii
 return inter/union if union else 1.
for directory in sorted(root.iterdir()):
 if directory.name=='views_0':continue
 other=load(directory.name);records=[]
 for video in videos:
  a=[x for x in base if x['video_id']==video];b=[x for x in other if x['video_id']==video]
  sims=np.array([[tube_iou(x,y) for y in b] for x in a]).reshape(len(a),len(b));class_mismatch=np.array([[x['category_id']!=y['category_id'] for y in b] for x in a]).reshape(len(a),len(b))
  row,col=linear_sum_assignment(1-sims+2*class_mismatch)
  matched=[(i,j) for i,j in zip(row,col) if not class_mismatch[i,j]]
  matches=[{'reference_index':int(i),'arm_index':int(j),'tube_iou_with_reference':float(sims[i,j]),'score_delta':float(b[j]['score']-a[i]['score']),'binary_masks_identical':a[i]['segmentations']==b[j]['segmentations'],'category_id':a[i]['category_id']} for i,j in matched]
  records.append({'video_id':video,'reference_tracks':len(a),'arm_tracks':len(b),'same_class_matches':len(matches),'matches':matches,'unmatched_reference':len(a)-len(matches),'unmatched_arm':len(b)-len(matches)})
 all_matches=[x for v in records for x in v['matches']];out.append({'arm':directory.name,'mean_tube_iou_with_reference':float(np.mean([x['tube_iou_with_reference'] for x in all_matches])) if all_matches else None,'minimum_tube_iou_with_reference':min((x['tube_iou_with_reference'] for x in all_matches),default=None),'binary_masks_changed_count':sum(not x['binary_masks_identical'] for x in all_matches),'mean_abs_score_change':float(np.mean([abs(x['score_delta']) for x in all_matches])) if all_matches else None,'per_video':records})
payload={'reference_arm':'views_0','scope':'unlabeled prediction sensitivity relative to matched no-update two-view outputs; these IoUs compare predictions to predictions, not GT, and are not task accuracy','methods':'detached same-predicted-class Hungarian track matching by RLE-computed full-tube IoU; retain unmatched counts','results':out};(study/'UNLABELED_VIEWS_SENSITIVITY.json').write_text(json.dumps(payload,indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='per_video'} for x in out],indent=2))
