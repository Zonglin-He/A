"""Check or snapshot the maintained public source and saved evidence, without inference."""
import argparse,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRIVATE={'.git','.conda','.venv','outputs','runtime','checkpoints','data','__pycache__','.pytest_cache'}
EXT={'.py','.md','.json','.txt','.yml','.yaml','.ini','.cpp','.cu','.cuh','.h','.sh','.tsv','.csv'}

def files():
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if any(x in PRIVATE or x.startswith('.venv')for x in rel.parts)or p.is_symlink()or not p.is_file():continue
  if rel.as_posix() in ['docs/sources.json','docs/publication.json']:continue
  if p.suffix in EXT or p.name in {'.gitignore','LICENSE'}:yield p

def inventory():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files()}
def check():
 cfg=json.loads((ROOT/'configs/default.json').read_text());assert cfg['temporal']['default']=='target_k16'
 assert all(d['temporal_mode']=='target_k16'for d in cfg['datasets'].values())
 assert not (ROOT/'scripts').exists()
 assert {p.name for p in (ROOT/'configs').glob('*.json')}=={'default.json'}
 for p in (ROOT/'stvg').glob('*.py'):
  s=p.read_text();assert not re.search(r'^(?:from|import) scripts[. ]',s,re.M),p
 for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://'in link or link.startswith('#'):continue
   assert (p.parent/link.split('#')[0]).exists(),(p,link)
 source=json.loads((ROOT/'docs/sources.json').read_text());assert source['files']==inventory(),'Run snapshot after an intentional source change'
 print(f"Archive check passed: {len(source['files'])} public source/evidence files; no model run")

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['check','snapshot']);a=p.parse_args()
 if a.command=='snapshot':
  (ROOT/'docs/sources.json').write_text(json.dumps(dict(schema='mainline_source_inventory_v1',archive_commit='c310af5888dca78d6ea7b0a280cb67d209a90337',files=inventory()),indent=2)+'\n');print('Source snapshot saved; no inference')
 else:check()
if __name__=='__main__':main()
