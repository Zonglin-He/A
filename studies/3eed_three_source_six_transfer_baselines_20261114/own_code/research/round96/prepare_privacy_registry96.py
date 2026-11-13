"""Extend the private publication deny registry with already-authorized nonprotected input identities."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round96';P=H/'publication';rd=lambda p:json.loads(p.read_text());v=rd(R/'research/round76/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json');tokens={(x['kind'],x['value']) for x in v['tokens']};cache={};n=0
for f in sorted((H/'experiment').glob('OFFICIAL_*_ORDER_PRIVATE.json')):
 for row in rd(f):
  path=row['original_meta_path'];tokens.add(('scene',row['scene']));tokens.add(('qualified_path',path))
  if path not in cache:cache[path]=rd(Path(path))
  cap=cache[path]['ground_info'][row['object_index']]['caption'];assert hashlib.sha256(cap.encode()).hexdigest()==row['caption_sha256'];tokens.add(('caption',cap));n+=1
out={'state':'R96_PRIVATE_NONPROTECTED_SOURCE_TRAIN_AND_EVAL_TOKEN_REGISTRY','tokens':[{'kind':k,'value':v} for k,v in sorted(tokens)],'registered_expression_references':n,'protected_media_opened':False};p=P/'PRIVATE_PRIVACY_TOKEN_REGISTRY.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print({'private_registry_tokens':len(tokens),'input_references':n,'raw_values_printed':False})
