"""Recreate local no-label manifests from the official archive-order cohort.
Requires the official extracted JPEG/FEATAudios files and full official checkpoint.
Redacted exported protocol receipts are evidence; regenerate a local locked plan.
"""
import pathlib,argparse,tarfile,json,ast,hashlib,subprocess,sys,numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--dataset-root',required=True);p.add_argument('--checkpoint',required=True);args=p.parse_args();data=pathlib.Path(args.dataset_root).resolve();archive=pathlib.Path(args.archive).resolve()
selected=[]
with tarfile.open(archive,'r|gz') as tar:
 for member in tar:
  q=pathlib.PurePosixPath(member.name)
  if q.is_absolute() or '..' in q.parts or member.issym() or member.islnk():raise ValueError(member.name)
  if member.isdir() and len(q.parts)==4 and q.parts[:3]==('datasets','test','JPEGImages') and not q.parts[3].startswith('.'):
   name=q.parts[3]
   if name not in selected:selected.append(name)
   if len(selected)==9:break
assert len(selected)==9;selected=selected[:8]
source=ast.parse((ROOT/'upstream/avism/data/datasets/avis.py').read_text());cats=next(ast.literal_eval(x.value) for x in source.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='AVIS_CATEGORIES' for t in x.targets))
base=[]
for name in selected:
 frames=sorted((data/'test/JPEGImages'/name).glob('*.jpg'),key=lambda f:int(f.stem.rsplit('_',1)[-1]));assert frames;numbers=[int(f.stem.rsplit('_',1)[-1]) for f in frames];assert numbers==list(range(numbers[0],numbers[0]+len(numbers))) and numbers[0] in [0,1]
 with Image.open(frames[0]) as im:w,h=im.size
 audio=data/'test/FEATAudios'/(name+'.npy');a=np.load(audio);assert a.shape==(len(frames),128) and np.isfinite(a).all()
 base.append({'video_id':name,'image_paths':[str(x) for x in frames],'audio_path':str(audio),'height':h,'width':w,'length':len(frames),'original_length':len(frames),'frame_indices':list(range(len(frames))),'source_frame_numbers':numbers})
for study_name,limit,locker in [('avis_prefix_dev_v3',16,'lock_plan.py'),('avis_full_context_v2',None,'lock_plan_full_v2.py')]:
 study=ROOT/'studies'/study_name;study.mkdir(parents=True,exist_ok=False);videos=json.loads(json.dumps(base))
 if limit:
  for v in videos:
   n=min(limit,v['length']);v['image_paths']=v['image_paths'][:n];v['frame_indices']=v['frame_indices'][:n];v['source_frame_numbers']=v['source_frame_numbers'][:n];v['length']=n
 spec={'study':study_name,'videos':videos,'categories':cats,'id_kind':'source_filename','selection':'first8 official archive-order test JPEG directories, chronological integer frame suffixes','sampling_limitation':'download-order convenience cohort; not representative full benchmark','label_exposure':'no GT JSON or count labels read'}
 if limit is None:
  parent=ROOT/'studies/avis_prefix_dev_v3/PLAN_LOCK.json';spec['parent_short_plan_sha256']=hashlib.sha256(parent.read_bytes()).hexdigest();spec['context_control']='full complete source video sequence, no truncation'
 (study/'inputs.json').write_text(json.dumps(spec,indent=2));subprocess.run([sys.executable,str(ROOT/'code'/locker),'--manifest',str(study/'inputs.json'),'--checkpoint',str(pathlib.Path(args.checkpoint).resolve()),'--output',str(study/'PLAN_LOCK.json')],check=True)
print('Local no-label plans recreated; score only after every arm seals. No inference launched.')
