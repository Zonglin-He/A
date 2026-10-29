from pathlib import Path
import time,hashlib,json,zipfile,stat
R=Path(__file__).resolve().parent;p=R/'downloads/3eed_dataset.zip'
for _ in range(480):
 if p.stat().st_size==6724434217:break
 time.sleep(15)
else:raise TimeoutError('Full dataset download incomplete after2hours')
h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert h=='2ba7acd85b5240ec8acb6b9684a38deacafe873375fab4e3262183ef469fd809'
out=(R/'dataset_full').resolve();out.mkdir(exist_ok=True);count=0;size=0
with zipfile.ZipFile(p) as z:
 for item in z.infolist():
  if item.filename.startswith('__MACOSX/') or item.filename.endswith('.DS_Store') or item.is_dir():continue
  dest=(out/item.filename).resolve()
  if not dest.is_relative_to(out) or stat.S_ISLNK(item.external_attr>>16):raise RuntimeError('Unsafe dataset member')
  size+=item.file_size
  if size>12*1024**3:raise RuntimeError('Archive unexpectedly exceeds12GB')
  dest.parent.mkdir(parents=True,exist_ok=True)
  with z.open(item) as src,dest.open('wb') as f:
   while b:=src.read(1024*1024):f.write(b)
  count+=1
receipt={'path':str(p),'sha256':h,'bytes':p.stat().st_size,'official_match':True,'extracted_files':count,'extracted_bytes':size,'dataset_root':str(out/'3eed_dataset')};(R/'full_dataset_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
