import pathlib,json,hashlib,ast,numpy as np
from PIL import Image
r=pathlib.Path(__file__).resolve().parents[1];rule=r/'studies/avis_prefix_dev_v3/COHORT_RULE_LOCK.json';s=json.loads(rule.read_text());assert hashlib.sha256(rule.read_bytes()).hexdigest()==rule.with_suffix('.json.sha256').read_text().split()[0]
seen=[]
for line in (r/'logs/extract_prefix.log').read_text().splitlines():
 parts=line.split()
 if len(parts)!=2:continue
 p=pathlib.PurePosixPath(parts[1])
 if len(p.parts)==4 and p.parts[:3]==('datasets','test','JPEGImages') and p.parts[3] in s['population_filenames'] and p.parts[3] not in seen:seen.append(p.parts[3])
assert len(seen)>=9,'Wait for first 8 source JPEG video directories to finish; next directory bounds completeness'
selected=seen[:8];assert selected==s['same_selected_filenames']
source=ast.parse((r/'upstream/avism/data/datasets/avis.py').read_text());cats=next(ast.literal_eval(x.value) for x in source.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='AVIS_CATEGORIES' for t in x.targets))
videos=[]
for name in selected:
 folder=r/'datasets/test/JPEGImages'/name;all_frames=sorted(folder.glob('*.jpg'),key=lambda p:int(p.stem.rsplit('_',1)[-1]));assert all_frames
 numbers=[int(p.stem.rsplit('_',1)[-1]) for p in all_frames];assert numbers==list(range(numbers[0],numbers[0]+len(numbers))) and numbers[0] in [0,1],(name,numbers)
 frames=all_frames[:16]
 for file in frames:
  with Image.open(file) as im:im.verify()
 with Image.open(frames[0]) as im:w,h=im.size
 audio=r/'datasets/test/FEATAudios'/(name+'.npy');partial=r/'datasets/test/FEATAudios'/(name+'.prefix16.npy')
 if not audio.exists():audio=partial
 a=np.load(audio);assert audio.name.endswith('.prefix16.npy') or a.shape[0]==len(all_frames),(audio,a.shape,len(all_frames));assert a.ndim==2 and a.shape[0]>=len(frames) and a.shape[1]==128 and np.isfinite(a[:len(frames)]).all(),(audio,a.shape)
 videos.append({'video_id':name,'image_paths':[str(x) for x in frames],'audio_path':str(audio),'height':h,'width':w,'length':len(frames),'original_length':len(all_frames),'frame_indices':list(range(len(frames))),'source_frame_numbers':numbers[:len(frames)]})
out=rule.parent/'inputs.json';assert not out.exists();out.write_text(json.dumps({'study':s['study'],'selection':s['selection'],'sampling_limitation':s['sampling_limitation'],'cohort_rule_lock_sha256':hashlib.sha256(rule.read_bytes()).hexdigest(),'videos':videos,'categories':cats,'id_kind':'source_filename','label_exposure':'no GT JSON or count labels read'},indent=2));print(out,hashlib.sha256(out.read_bytes()).hexdigest())
