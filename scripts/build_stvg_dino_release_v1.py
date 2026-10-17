"""Materialize a code-only import closure for the restored P7-A entrypoint.

Copies only existing byte-identical source files. Never deletes workspace files
or executes a scientific stage. Third-party data/weights remain external.
"""
from pathlib import Path
import ast,collections,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def build():
 from scripts import build_stvg_main_release_v1 as h
 roots=['methods/stvg_opd_p7a_v1/predictor.py','scripts/verify_stvg_dino_rollback_v1.py','scripts/report_stvg_dino_rollback_v1.py','scripts/build_stvg_dino_release_v1.py','scripts/research_archive.py']
 queue=collections.deque(ROOT/x for x in roots);files={};edges={};missing=set()
 while queue:
  p=queue.popleft().resolve();rel=str(p.relative_to(ROOT))
  if rel in files:continue
  raw=p.read_bytes();tree=ast.parse(raw,filename=rel);files[rel]=dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());deps=set()
  for parent in p.parents:
   if parent==ROOT:break
   q=parent/'__init__.py'
   if q.exists()and q!=p:deps.add(q.resolve())
  for module,kind in h.imported_modules(tree,p):
   candidates=h.module_candidates(module)
   deps.update(candidates)
   if not candidates and kind!='from-submodule'and module.split('.')[0]in h.SOURCE_DIRS:missing.add(module)
  for _,q in h.literal_local_py_paths(tree,p):deps.add(q.resolve())
  edges[rel]=sorted(str(q.relative_to(ROOT))for q in deps);queue.extend(sorted(deps))
 out=ROOT/'dino_method_release';dest=out/'code';dest.mkdir(parents=True,exist_ok=True)
 for rel,v in files.items():
  p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
  else:shutil.copy2(ROOT/rel,p)
 manifest=dict(status='code_materialized',scope='restored DINO P7-A online Gaussian OPD, original frozen Actionness projection; no Qwen continuation',roots=roots,source_file_count=len(files),source_files=[files[k]for k in sorted(files)],internal_edges=edges,unresolved_static_module_names=sorted(missing),known_external_assets='original source checkpoints, Grounding DINO snapshot, model_zoo, official inputs and compatible tubedetr environment; not public',workspace_files_deleted=0,new_model_or_GT_calls=0)
 p=out/'DEPENDENCY_MANIFEST.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items()if k not in ['source_files','internal_edges']}))
 return manifest
if __name__=='__main__':build()
