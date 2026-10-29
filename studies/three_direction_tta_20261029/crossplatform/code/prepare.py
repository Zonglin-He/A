import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
names=json.loads((ROOT/'downloads/subset_names.json').read_text());samples=[]
for name in names:
 if not name.endswith('meta_info.json'):continue
 p=ROOT/'assets'/name
 meta=json.loads(p.read_text());plat=name.split('/')[1]
 # One fixed query per independent frame; no correctness or class selection.
 if not meta['ground_info']:raise RuntimeError('Empty frame')
 obj=meta['ground_info'][0]
 samples.append({'id':hashlib.sha256(name.encode()).hexdigest()[:16],'platform':plat,'meta_path':str(p),'lidar_path':str(p.with_name('lidar.npy' if plat=='waymo' else 'lidar.bin')),'text':' '.join(obj['caption'].replace(',',' ,').split())+' . not mentioned','obj_index':0})
(ROOT/'manifest.json').write_text(json.dumps(samples,indent=2))
print('samples',len(samples))
