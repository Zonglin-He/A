"""Full-target pinned Tent/SAR extension; all previous scientific files immutable."""
import os,sys,json,time,hashlib,csv,importlib.util,types
from pathlib import Path
from functools import lru_cache
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/no_shrink_main_v1'),str(R),str(R/'tools')]
import ns_common as ns
from full import read,sha,atomic,identity,check,RUN,PROVIDER,binding
D=R/'outputs/tta_full_v1';P=R/'results/tta_full_v1';K0=ns.D;PREV=ns.PREV;OLD=ns.OLD;COUNTS=ns.COUNTS
fixed=lru_cache(None)(read);digest=ns.digest;records=ns.records;labels=ns.labels;exited=ns.exited
@lru_cache(None)
def baseline():return ns.load_old('_pinned_full_TTA_parent',R/'studies/map_paper_completion_v1/tta.py',ns.suite())
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(f)==h,f
 check();return p
def parent(ds,row):return ns.suite().load_parent(ds,row)
def reused(ds,key):return fixed(D/'reuse_index.json')[ds].get(key)
def contexts(ds,key):return fixed(D/'context_index.json')[ds].get(key)
def payload_context(ds,key):
 import torch
 e=contexts(ds,key);assert e is not None;assert sha(e['path'])==e['sha256'];v=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);c=v['temporal']if e['kind']=='shared_native'else v;assert not c['GT_read'];return c,e

def check_context(c,cap):
 import torch
 assert c['schema']=='ta_native224_time_decoder_inputs_v1'and c['frame_ids']==cap['frame_ids']and not c['GT_read'];assert len(c['offsets'])==2
 for i,x in enumerate(c['offsets']):
  assert x['frame_ids']==cap['frame_ids'][i::2]and torch.equal(x['native_logits'],cap['native_logits'][i]);assert set(x['inputs'])=={'query_tgt','query_content','query_time','query_mask','encoded_feature','encoded_pos','encoded_mask'}
  assert set(x['inputs'])==set(x['input_layouts'])
  for k,v in x['inputs'].items():assert list(v.shape)==x['input_layouts'][k]['shape']

def entry_files(ds,key):
 dest=D/ds;return dest/'predictions'/(key+'.npz'),dest/'adapted_logits'/(key+'.pt'),dest/'receipts'/(key+'.json')
def existing_entry(ds,key):
 out,lp,rp=entry_files(ds,key);assert rp.exists();e=read(rp)
 for f,h in [(out,'sha256'),(lp,'logits_sha256')]:assert sha(f)==e[h]
 return dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
def seal(ds):
 p=validate();files={r['id']:existing_entry(ds,r['id'])for r in records(ds)};assert len(files)==COUNTS[ds];counts=dict(queries=len(files),configs=2*len(files),reused_completed_queries=sum(e['reused_completed']for e in files.values()),new_optimizer_iterations=sum(e['actual_new_counts']['optimizer_iterations']for e in files.values()),new_Source_queries=sum(e['actual_new_counts']['new_Source_queries']for e in files.values()),shared_Source_queries=sum(e['actual_new_counts']['shared_Source_queries']for e in files.values()),new_expert=0);atomic(D/f'{ds}_sealed.json',dict(files=files,counts=counts,queries=len(files),fingerprint=p['fingerprint'],GT_read=False));atomic(D/f'{ds}_prediction_state.json',dict(status='complete_sealed',counts=counts))
def guard_resources():
 import shutil
 p=read(D/'plan.json');assert shutil.disk_usage(R).free>=p['limits']['minimum_disk_free_bytes'];assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['limits']['maximum_private_bytes']
