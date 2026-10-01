"""CPU parses and exact previous-expert receipt index; no label/model use."""
import sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_full_common_v1 import *
def signature(row):
 q=row['input'];return __import__('hashlib').sha256(__import__('json').dumps({k:q[k] for k in ['caption','source','video_sha256','frame_ids','width','height','fps']},sort_keys=True).encode()).hexdigest()
def run(dataset):
 p=verify(dataset);out=BASE/dataset
 if (out/'SUBJECT_BARRIER.json').exists():return
 origins=[]
 if dataset=='vidstg':
  b=ROOT/'artifacts/tastvg_full_b1_v1';origins.append((b,read(b/'ROSTER_LOCK.json')['rows']))
 paper=ROOT/'artifacts/tastvg_paper48_v1';b=paper/('P1' if dataset=='vidstg' else 'P5');origins.append((b,read(b/'PLAN.json')['rows']))
 for name in ['tastvg_optuna_v1','tastvg_coordinate_tuning_v2','tastvg_extended_sensitivity_v3']:
  b=ROOT/'artifacts'/name/dataset;origins.append((b,read(b/'PLAN.json')['rows']))
 subjects={};experts={}
 for b,rows in origins:
  for i,r in enumerate(rows):
   j=r.get('ordinal',i);sf=b/'subjects'/f'{j:05}.json'
   if sf.exists():
    rr=read(sf);subjects[rr['caption_sha256']]=sf
   # P1 specialists live at parent/expert, indexed by original B1/Paper48 rows.
   eb=paper/'experts' if b==paper/'P1' else b/'experts'
   experts.setdefault(signature(r),[]).append((str(eb),j))
 needed=read(out/'EXPERT_PLAN.json')['expert_needed'];index={}
 for i in needed:
  for stage in ['spatial','temporal']:
   for cond in p['conditions']:
    for folder,j in reversed(experts.get(signature(p['rows'][i]),[])):
     f=Path(folder)/stage/cond/f'{j:05}.json'
     if f.exists():index[f'{stage}/{cond}/{i:05}.json']=dict(receipt=str(f),receipt_sha256=sha(f),folder=folder);break
 write(out/'EXPERT_REUSE_INDEX.json',index)
 parser=None;files={};new=0
 for i,r in enumerate(p['rows']):
  f=out/'subjects'/f'{i:05}.json';cap=hashlib.sha256(r['input']['caption'].encode()).hexdigest()
  if not f.exists():
   if cap in subjects:x=read(subjects[cap]);x={**x,'ordinal':i}
   else:
    if parser is None:
     import torch
     torch.set_num_threads(4)
     from vg_tta.foreground_runtime import QuerySubjectParser
     parser=QuerySubjectParser(ROOT/'.cache/stanza')
    x=dict(ordinal=i,caption_sha256=cap,parses=parser(r['input']['caption']));new+=1
   write(f,x)
  assert read(f)['caption_sha256']==cap;files[f.name]=sha(f)
  if i%100==0:status(out/'PREPARATION_STATUS.json',dict(status='parsing',done=i,total=len(p['rows']),new_parses=new));print('SUBJECT',dataset,i,len(p['rows']),flush=True)
 write(out/'SUBJECT_BARRIER.json',dict(count=len(p['rows']),files=files,GT_read=False,expert_reuse_index_sha256=sha(out/'EXPERT_REUSE_INDEX.json'),time=time.time()))
if __name__=='__main__':run(sys.argv[1])
