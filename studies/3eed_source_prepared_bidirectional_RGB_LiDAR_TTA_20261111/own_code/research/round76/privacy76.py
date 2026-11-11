"""Extend the existing private-token scan to newly used Waymo source media."""
from common76 import *
base=read(ROOT/'research/round69/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json');seen={(x['kind'],x['value']) for x in base['tokens']};out=list(base['tokens'])
for r in source_rows('TRAIN512')+source_rows('VAL128')+target_rows():
 for field,kind in [('id','id'),('alias','alias'),('scene','scene'),('caption','caption'),('pointpath','media_path'),('imagepath','media_path'),('metapath','media_path')]:
  v=r.get(field)
  if v and (kind,v) not in seen:out.append({'kind':kind,'value':v});seen.add((kind,v))
p=H/'publication';p.mkdir(exist_ok=True);dump(p/'PRIVATE_PRIVACY_TOKEN_REGISTRY.json',{'state':'R76_PRIVATE_RAW_TOKEN_REGISTRY_EXPLICIT_INPUTS','tokens':out,'extends_prior_registry':True,'source_inputs_added':640,'coverage':{**base['coverage'],'new_source_selected_ids':640},'no_new_FINAL_file_access':True});print({'tokens':len(out),'source_added':640})
