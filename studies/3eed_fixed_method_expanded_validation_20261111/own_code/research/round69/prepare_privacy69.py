"""Extend the saved private-token registry with the new fixed cohorts."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round69'
def rd(p):return json.loads(Path(p).read_text())
base=R/'research/round50/final_inputs_attempt001/PRIVATE_PRIVACY_TOKEN_REGISTRY.json';old=rd(base);rows=rd(H/'input/ALL2304_PRIVATE.json');out=H/'publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json';assert not out.exists();by={(t['kind'],t['value']):t for t in old['tokens']}
for r in rows:
 for k,v in [('id',r['id']),('scene',r['scene']),('caption',r['caption']),*[("media_path",r[k]) for k in ['pointpath','metapath','imagepath']]]:
  assert isinstance(v,str) and v;by[k,v]={'kind':k,'value':v}
cov=dict(old['coverage']);cov.update(R69_selected_input_ids=2304,R69_train_input_ids=1024,R69_val_input_ids=1280,R69_all_selected_captions_and_media_paths_scanned=True,base_registry_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),registry_builder_opens_only_saved_registry_and_selected_nonFINAL_manifest=True)
out.parent.mkdir(exist_ok=True);out.write_text(json.dumps({'tokens':list(by.values()),'coverage':cov},indent=2)+'\n');print({'tokens':len(by),'new_tokens':len(by)-len(old['tokens']),'coverage':cov})
