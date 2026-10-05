"""User-authorized exact inactive weights/transient caches; retained predictions."""
import json,os,time,shutil,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/storage_cleanup_20261005'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def identity(p):
 s=p.stat();return dict(dev=s.st_dev,inode=s.st_ino,bytes=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,links=s.st_nlink)
def run():
 assert not (OUT/'DELETE_MANIFEST.json').exists();OUT.mkdir(parents=True,exist_ok=True)
 from scripts.cleanup_unused_model_weights_20261001 import open_file_check
 files=[]
 sa=ROOT/'checkpoints/Sa2VA-4B';rec=json.loads((sa/'DOWNLOAD_RECEIPT.json').read_text())
 for x in rec['files']:
  p=sa/x['file'];assert p.is_file() and not p.is_symlink()
  assert p.stat().st_size==x['bytes']
  files.append(dict(path=str(p),category='retired_Sa2VA_official_weights',historical_sha256=x['sha256'],download_receipt=str(sa/'DOWNLOAD_RECEIPT.json'),**identity(p)))
 p=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors';r=p.parent/'OFFICIAL_RECEIPT.json';rec=json.loads(r.read_text())
 assert str(p)==rec['path'] and p.stat().st_size==rec['bytes'] and not p.is_symlink()
 files.append(dict(path=str(p),category='retired_PTD_official_weights',historical_sha256=rec['sha256'],download_receipt=str(r),**identity(p)))
 paper=ROOT/'artifacts/tastvg_paper48_v1';assert (paper/'FINAL_COMPLETION.json').is_file()
 # This explicitly transient encoder cache duplicates Frozen predictions already
 # retained in sealed P0--P5 payloads; no sealed online artifact is removed.
 for p in sorted((paper/'scratch_H').glob('*.pt')):
  assert not p.is_symlink();files.append(dict(path=str(p),category='completed_Paper48_transient_H',**identity(p)))
 pip=Path('/home/wwww/.cache/pip')
 for p in sorted(pip.rglob('*')):
  if p.is_file() and not p.is_symlink():files.append(dict(path=str(p),category='rebuildable_pip_download_cache',**identity(p)))
 protected=[]
 for folder in ['.cache/grounding_dino_http_v1','.cache/huggingface','.cache/torch','.cache/stanza','checkpoints/tastvg_model_zoo']:
  protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
 protected.extend(p for p in (ROOT/'checkpoints').glob('TASTVG_*.pth') if p.is_file())
 protected.extend([ROOT/'artifacts/decota_fixed_full_corruption_v1/RUNTIME_LOCK.json',ROOT/'methods/CURRENT_METHOD.json'])
 before={str(p):identity(p) for p in protected};assert not set(before)&{x['path'] for x in files}
 assert all(x['links']==1 for x in files),'Linked assets must be reviewed before any deletion'
 opened=open_file_check(files)
 free=shutil.disk_usage(ROOT).free
 write(OUT/'DELETE_MANIFEST.json',dict(time=time.time(),user_authorization='Delete previous unused caches, model weights and other unused assets',
  files=files,protected_stat=before,open_file_check=opened,free_before=free,
  prediction_retention='Every sealed experiment payload, report, negative result and failure record retained; only explicit transient H duplicates retired',
  restoration='Official weights can be restored using preserved download repository/revision/SHA receipts; old specialist replay needs restoration'))
 completed=[]
 for x in files:
  p=Path(x['path']);assert identity(p)=={k:x[k] for k in identity(p)},p
  assert x['links']==1,'Linked asset retained for manual accounting'
  # Official SHA was verified by the retained intake receipt; bytes and inode
  # identity are checked again immediately before unlinking.
  p.unlink();completed.append(x)
  write(OUT/'PROGRESS.json',dict(done=len(completed),total=len(files),time=time.time()))
 assert all(identity(Path(p))==v for p,v in before.items())
 from scripts.decota_fixed_full_common_v1 import verify,archive
 verify();after=shutil.disk_usage(ROOT).free
 result=dict(status='completed',files=len(completed),logical_deleted_bytes=sum(x['bytes'] for x in completed),allocated_deleted_bytes=sum(x['allocated'] for x in completed),
  net_free_increase_bytes=after-free,free_before=free,free_after=after,protected_unchanged=True,active_runtime_verified=True,
  categories={k:sum(x['allocated'] for x in completed if x['category']==k) for k in sorted({x['category'] for x in completed})},time=time.time())
 write(OUT/'COMPLETION.json',result)
 note=ROOT/'docs/STORAGE_CLEANUP_20261005.md'
 note.write_text('# Inactive assets retired on 2026-10-05\n\n'
  f'User-authorized deletion completed: {len(completed)} files; allocated storage retired {result["allocated_deleted_bytes"]/2**30:.3f} GiB; '
  f'measured net free increase {result["net_free_increase_bytes"]/2**30:.3f} GiB, available {after/2**30:.3f} GiB.\n\n'
  'Removed the unused official Sa2VA-4B and PTD-Qwen3-VL-4B weight files, the completed Paper48 transient scratch_H cache and rebuildable pip downloads. '
  'Configs, source code, tokenizer/index files and original official download/revision/SHA receipts are retained; exact old teacher replay requires restoring weights first. '
  'All sealed online predictions, scientific negative results, reports, failure records, trained checkpoints, official datasets and installed environments remain present.\n\n'
  'The full DeCoTA dependencies (both TA-STVG checkpoints, Grounding DINO, RoBERTa, ResNet/Swin and Stanza), current production registration and runtime pins matched before/after. '
  'Selected file inode/stat identity was checked immediately before deletion; accessible process fds/maps showed no selected file in use. Permission-limited system processes are outside that check. '
  'Deletion manifests and actual receipts are in artifacts/storage_cleanup_20261005. Full evaluation continues independently.\n')
 archive(f'用户授权不用资产清理实际完成：释放{result["net_free_increase_bytes"]/2**30:.3f}GiB，现可用{after/2**30:.3f}GiB；当前依赖与运行pin未变，科学预测/失败证据保留')
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()
