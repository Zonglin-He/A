"""Fixed text-only cohort. Never reads metadata, algorithms or task outcomes."""
from pathlib import Path
import json,re,hashlib,datetime
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;P=R/'research/round24/input/ALL336_PRIVATE.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads(P.read_text());assert len(rows)==336
bins={(pl,st):[] for pl in ['drone','quad'] for st in ['explicitObjecttype','freeform']}
for x in rows:
 # Only these three fields are used. Do not inspect GT class, boxes or paths.
 r={k:x[k] for k in ['id','platform','caption']};st='explicitObjecttype' if re.search(r'\bobject\s+type\s*:',r['caption'],re.I) else 'freeform';r['stratum']=st;r['selection_sha256']=hashlib.sha256(('R48blind|'+r['id']).encode()).hexdigest();bins[r['platform'],st].append(r)
selected=[];counts={}
for (pl,st),part in bins.items():
 part.sort(key=lambda r:(r['selection_sha256'],r['id']));counts[pl+'|'+st]={'available':len(part),'selected':min(6,len(part))};selected.extend(part[:6])
for i,r in enumerate(selected,1):r['review_alias']=f'T{i:02d}'
cohort=D/'COHORT_PRIVATE.json';assert not cohort.exists();cohort.write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n');lock={'state':'FIXED_TEXT_ONLY_BLIND_COHORT_BEFORE_ANNOTATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'research/round24/input/ALL336_PRIVATE.json','source_manifest_sha256':sha(P),'fields_read_for_selection':['id','platform','caption'],'explicit_field_rule':r'\bobject\s+type\s*: case-insensitive','selection':"lowest6 sha256('R48blind|'+id) independently in eachplatformxstratum; no GT or algorithm outputs",'stratum_counts':counts,'n':len(selected),'cohort_sha256':sha(cohort),'no_algorithm_output_GTclass_bbox_IoU_media_XYZ_metadata_FINAL_access':True,'nonexpert_text_role_review_not_gold':True};(D/'COHORT_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(lock,indent=2))
