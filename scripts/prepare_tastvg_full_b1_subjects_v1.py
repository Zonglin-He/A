import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_full_b1_v1'
if __name__=='__main__':
 import torch
 from vg_tta.foreground_runtime import QuerySubjectParser
 torch.set_num_threads(4);p=read(OUT/'ROSTER_LOCK.json');parser=QuerySubjectParser(ROOT/'.cache/stanza');tick=time.time()
 for i,r in enumerate(p['rows']):
  f=OUT/'subjects'/f'{i:05}.json'
  if not f.exists():write(f,dict(ordinal=i,caption_sha256=__import__('hashlib').sha256(r['input']['caption'].encode()).hexdigest(),parses=parser(r['input']['caption'])))
  if i%200==0:print('subjects',i,len(p['rows']),flush=True)
 write(OUT/'SUBJECT_BARRIER.json',dict(files={f.name:sha(f) for f in (OUT/'subjects').glob('*.json')},count=len(p['rows']),seconds=time.time()-tick,GT_read=False))
