"""Human-authorized finite deletion of enumerated unused packages/models/rebuildable caches."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/storage_cleanup_20261017_unused_v2'
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def available():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
def opened():
 inodes=set();commands=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');commands.append((p.name,cmd))
   for f in (p/'fd').iterdir():
    try:s=f.stat();inodes.add((s.st_dev,s.st_ino))
    except OSError:pass
   for line in (p/'maps').read_text().splitlines():
    x=line.split(None,5)
    if len(x)==6 and x[5].startswith('/'):
     try:s=Path(x[5]).stat();inodes.add((s.st_dev,s.st_ino))
     except OSError:pass
  except (OSError,PermissionError):pass
 return inodes,commands
def main():
 assert not BASE.exists();targets=[]
 def add(p,category,reason):
  p=p.resolve();assert p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink();s=p.stat();targets.append(dict(path=str(p.relative_to(ROOT)),category=category,reason=reason,bytes=s.st_size,allocated=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,nlink=s.st_nlink))
 for folder,pattern,category,reason in [
  ('downloads/dataset_intake_20260910/datasets/MeViSv2','**/*.tar','unused_dataset_packages','unselected segmentation dataset download; metadata/images retained'),
  ('downloads/dataset_intake_20260910/datasets/MeViSv2','sample_submission_valid.zip','unused_dataset_packages','unused sample-submission package'),
  ('downloads/dataset_intake_20260910/datasets/SportsGrounding','*.zip','unused_dataset_packages','old optional training/validation archives; extracted research media and evidence retained'),
  ('downloads/dataset_intake_20260910/datasets/VidOR','training-video-part*.zip','unused_dataset_packages','optional training archives not consumed by current HC/Vid evaluation; original evaluation source videos retained'),
  ('downloads/dataset_intake_20260910/datasets/GroundingYouTube/videos','*.tar','unused_dataset_packages','old archive shards; extracted clips/frames and evidence retained'),
  ('checkpoints/VideoChat-R1_5-7B','model-*.safetensors','inactive_model_weights','not CURRENT, paper or current R1-only/conditional DINO-R1 checkpoint; model identity and official download receipts retained'),
  ('checkpoints','STCAT_hc_res416.partial.range*','abandoned_download','obsolete incomplete STCAT download fragments; no complete model consumed'),
  ('artifacts/stvg_factorial_native256_v1/native_cache','*/native.npz','rebuildable_teacher_cache','old completed caption-fix factorial consumed cache; policy/Teacher outputs/Source/predictions retained'),
  ('artifacts/stvg_paper_t2_b_v1/native_cache','*/native.npz','rebuildable_teacher_cache','paper256 publicly closed; current exact baseline uses complete saved predictions, context NPY and Source prefix, not this Teacher encoding cache')]:
  for p in sorted((ROOT/folder).glob(pattern)):add(p,category,reason)
 assert targets and len({r['path'] for r in targets})==len(targets)
 for path in ['artifacts/stvg_factorial_native256_captionfix002/CPU_COMPLETION.json','artifacts/stvg_factorial_native256_captionfix002/publication/factorial256/PUBLICATION_COMPLETION.json','artifacts/stvg_paper_t2_b_v1/CPU_COMPLETION.json','artifacts/stvg_paper_t2_b_v1/publication/paper/PUBLICATION_COMPLETION.json']:
  assert json.loads((ROOT/path).read_text())['status']=='complete'
 protected=[ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json']+[ROOT/rel for name in ['artifacts/stvg_r1_geometry32_v1/EXECUTION_RUNTIME002.json','artifacts/stvg_r1_dino_complement32_v1/EXECUTION_RUNTIME.json'] if (ROOT/name).exists() for rel in json.loads((ROOT/name).read_text())['pins']]
 selected={r['path'] for r in targets};assert not any(str(p.relative_to(ROOT)) in selected for p in protected);protected={str(p.relative_to(ROOT)):digest(p) for p in protected};before=available();busy,commands=opened();assert not any((r['device'],r['inode']) in busy for r in targets)
 for _,cmd in commands:assert not any(str(ROOT/r['path']) in cmd for r in targets)
 def bind(r):return dict(r,sha256=digest(ROOT/r['path']))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:targets=list(pool.map(bind,targets))
 # Native cache SHA is bound by its preserved original POLICIES receipt.
 for r in targets:
  if r['category']=='rebuildable_teacher_cache':assert json.loads((ROOT/r['path']).with_name('POLICIES.json').read_text())['native_sha256']==r['sha256']
 busy,_=opened();assert not any((r['device'],r['inode']) in busy for r in targets);write(BASE/'AUDIT_LOCK.json',dict(status='locked',records=targets,protected=protected,open_fd_mmap_checks=2,explicit_human_cleanup_authorization=True,GT_read=False))
 deleted=[]
 for r in targets:
  p=ROOT/r['path'];s=p.stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==[r[k] for k in ['device','inode','bytes','mtime_ns']];p.unlink();deleted.append(r['path'])
  if r['category']=='rebuildable_teacher_cache':write(p.with_name('NATIVE_CACHE_RETIRED_20261017.json'),dict(status='retired',original_path=r['path'],original_sha256=r['sha256'],bytes=r['bytes'],reason=r['reason'],recipe_and_pixels_retained=True,immediate_tensor_replay_requires_rebuild=True,GT_read=False))
 for p,h in protected.items():assert digest(ROOT/p)==h,p
 assert all(not (ROOT/p).exists() for p in deleted);after=available();summary={}
 for r in targets:
  x=summary.setdefault(r['category'],dict(count=0,logical_bytes=0,allocated_bytes=0));x['count']+=1;x['logical_bytes']+=r['bytes'];x['allocated_bytes']+=r['allocated']
 write(BASE/'COMPLETION.json',dict(status='complete',count=len(targets),categories=summary,logical_bytes=sum(r['bytes'] for r in targets),allocated_bytes=sum(r['allocated'] for r in targets),available_before=before,available_after=after,net_available_increase=after-before,active_protected_hashes_pass=True,current_Student_Source_R1_DINO_weights_and_HC_Vid_primary_sources_preserved=True,predictions_Teacher_responses_and_all_reports_preserved=True,retired_model_and_data_require_redownload=True,GT_read=False));print(json.dumps(json.loads((BASE/'COMPLETION.json').read_text())))
if __name__=='__main__':main()
