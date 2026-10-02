"""Explicit allow-list public manifest; does not upload or read private assets."""
import sys,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/tastvg_event_support_v1'
PUBLIC=Path('/home/wwww/visual-grounding-public-A')
def git(*args):return subprocess.check_output(['git',*args],cwd=PUBLIC,text=True).strip()
def run():
 export=json.loads((BASE/'ROOT_EXPORT.json').read_text());assert export['total_arrivals']==1536
 folder=PUBLIC/export['result_relative_path'];paths=set(export['code_paths'])|{'docs/TA_EVENT_SUPPORT_REVIEW.md'}
 paths|={str(p.relative_to(PUBLIC)) for p in folder.rglob('*') if p.is_file()}
 for rel in paths:assert Path(rel).parts[0] in ['scripts','vg_tta','protocols','docs','results'] and not any(part in ['data','weights','downloads','online','capture','experts'] for part in Path(rel).parts)
 parent=git('rev-parse','origin/main');assert parent==git('rev-parse','HEAD') and not git('diff','--cached','--name-only')
 remote_blobs={line.split()[2] for line in git('ls-tree','-r',parent).splitlines()}
 files=[]
 for rel in sorted(paths):
  p=PUBLIC/rel;b=p.read_bytes();blob=git('hash-object',str(p));reuse=blob in remote_blobs
  text=p.suffix not in ['.pdf','.png'];content=b.decode('utf-8') if text else None
  files.append(dict(path=rel,bytes=len(b),characters=len(content) if text else None,sha256=hashlib.sha256(b).hexdigest(),git_blob_sha=blob,reuse_existing_blob=reuse,encoding='utf-8' if text else 'base64'))
 manifest=dict(repository='Zonglin-He/A',branch='main',parent_sha=parent,base_tree_sha=git('rev-parse',parent+'^{tree}'),files=files,total_bytes=sum(x['bytes'] for x in files),file_count=len(files),public_result_path=export['result_relative_path'],privacy_check='explicit code/protocol/report/anonymous scalars/figures; excludes media, annotations, captions, private source identifiers, tensors, weights and conversation attachments')
 (BASE/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ['repository','parent_sha','file_count','total_bytes']},indent=2))
if __name__=='__main__':run()
