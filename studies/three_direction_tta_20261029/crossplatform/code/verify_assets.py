import hashlib,json,zipfile,stat
from pathlib import Path
R=Path(__file__).resolve().parent
metadata=json.loads((R/'downloads/official_asset_metadata.json').read_text());checks={}
for group,name,path in [('3eed','baseline_ckpt_pred.zip',R/'downloads/baseline_ckpt_pred.zip'),]:
 m=next(x for x in metadata[group] if x['path']==name)
 if path.stat().st_size!=m['size']:raise RuntimeError(f'Incomplete {path.name} {path.stat().st_size}/{m["size"]}')
 h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
 if h!=m['lfs']['oid']:raise RuntimeError('Official LFS SHA256 mismatch')
 checks[name]={'bytes':m['size'],'sha256':h,'official_match':True}
z=zipfile.ZipFile(R/'downloads/baseline_ckpt_pred.zip')
for name in ['final_6384/ckpt_6384.pth','final_6384/config.json']:
 info=z.getinfo(name)
 if stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('Unsafe symlink')
 dest=(R/'assets'/name).resolve()
 if not dest.is_relative_to(R/'assets'):raise RuntimeError('Unsafe extraction path')
 dest.parent.mkdir(parents=True,exist_ok=True)
 with z.open(name) as src,dest.open('wb') as out:
  while b:=src.read(4*1024*1024):out.write(b)
checks['checkpoint']={'sha256':hashlib.file_digest((R/'assets/final_6384/ckpt_6384.pth').open('rb'),'sha256').hexdigest()}
(R/'assets_verified.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
