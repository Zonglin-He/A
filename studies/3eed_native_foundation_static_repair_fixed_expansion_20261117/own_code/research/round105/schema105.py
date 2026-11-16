"""Inspect only archive schema and sensor-input counts; no media, models or GT."""
from pathlib import Path
import json,ijson
R=Path(__file__).resolve().parents[2];H=R/'research/round105';p=R/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json';x=json.loads(p.read_text());out={'old_frame_index_type':type(x).__name__,'old_frame_index_n':len(x)}
if isinstance(x,dict):
 out['top_level_fields']=list(x)[:15]
 for key,value in x.items():
  if isinstance(value,list) and value and isinstance(value[0],dict):out['record_container']=key;out['record_fields']=list(value[0]);out['records']=len(value);break
elif x and isinstance(x[0],dict):out['record_fields']=list(x[0])
protected=[]
with (R/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json').open('rb') as f:
 for prefix,event,value in ijson.parse(f):
  if prefix=='item.metapath' and event=='string':protected.append(str(Path(value).resolve()))
out['protected_identity_count_for_exclusion_only']=len(set(protected));out['protected_media_read']=False;out['GT_or_model_used']=False;(H/'SCHEMA.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
