"""Versioned adoption of the explicitly selected no-shrinkage method."""
import os,sys,json,time,csv,hashlib,importlib.util,types,inspect
from pathlib import Path
from functools import lru_cache,partial
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/map_full_v1'),str(R/'studies/paper_review_suite_v1'),str(R/'studies/paper_followup_v1')]
from full import RUN,PROVIDER,read,sha,atomic,identity,check,binding
D=R/'outputs/no_shrink_main_v1';P=R/'results/no_shrink_main_v1';PREV=R/'outputs/map_paper_completion_v1';OLD=R/'outputs/paper_review_suite_v1';MAP=R/'outputs/map_full_now_v3';CG=R/'outputs/cgstvg_full_v1'
COUNTS={'hc2':1902,'vidstg':10303};fixed=lru_cache(maxsize=None)(read)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def load_old(name,path,common=None):
 prev=sys.modules.get('common');
 if common is not None:sys.modules['common']=common
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 finally:
  if prev is None:sys.modules.pop('common',None)
  else:sys.modules['common']=prev
 return m
@lru_cache(None)
def suite():return load_old('_old_completion_common',R/'studies/map_paper_completion_v1/common.py')
@lru_cache(None)
def components():
 from studies.paper_followup_v1 import common
 return load_old('_old_components',R/'studies/map_components_v1/run.py',common)
@lru_cache(None)
def spatial():return load_old('_old_spatial_core',R/'studies/map_paper_completion_v1/spatial_core.py',suite())
def records(ds):return suite().records(ds)
def labels(ds):return suite().labels(ds)
def no_shrink(ds,key):
 seal=fixed(PREV/f'temporal_v2_{ds}_sealed.json');e=seal['files'][key];p=PREV/e['path'];rp=PREV/e['receipt_path'];assert sha(p)==e['sha256'] and sha(rp)==e['receipt_sha256'];return p,dict(path=str(p),sha256=e['sha256'],receipt=str(rp),receipt_sha256=e['receipt_sha256'])
def verify_main(ds,key,a):
 import numpy as np
 p,e=no_shrink(ds,key)
 with np.load(p)as z:assert np.array_equal(a['FullMAP'],z['NoShrink']) and np.array_equal(a['FullMAP_interval'],z['NoShrink_interval'])
 return e
def exited(w):
 try:s=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return s[0]=='Z' or int(s[19])!=w['startticks']
 except FileNotFoundError:return True
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(f)==h,f
 check();return p
def pairs(task):
 if task=='components':return [('EightReplacement','Source'),('SharedCorrection','EightReplacement'),('SpatialInference','SharedCorrection'),('FullMAP','SpatialInference'),('FullMAP','Source'),('FullMAP','SourceJointMAP'),('FullMAP','FinalIndependent')]
 if task=='budget':return [(a,'Source')for a in components().CONFIGS]+[(f'Hybrid{k}','Hybrid8')for k in [2,4,16]]+[(f'Uniform{k}',f'Hybrid{k}')for k in [2,4,8,16]]
 if task=='spatial':return [('FullMAP',a)for a in arms(task)if a not in ['FullMAP','S1T1']]+[('S1T1','S1T0'),('S0T1','S0T0'),('S1T0','S0T0')]
 return [('FullMAP','Source')]+([('FullMAP','SourceJointMAP')]if task=='CG'else[])
def arms(task):return components().COMPONENTS if task=='components' else ['Source',*components().CONFIGS]if task=='budget'else spatial().ARMS if task=='spatial'else ['Source','SourceJointMAP','FullMAP']if task=='CG'else ['Source','FullMAP']
def rows(task,ds):
 rr=records(ds)if task in ['components','CG','spatial']else fixed(OLD/'selection.json')[ds]['confirm']
 return [dict(row=r,key=r['id']+'.'+v,view=v)for r in rr for v in ['Blur','JPEG','Dim']]if task=='corruption'else[dict(row=r,key=r['id'])for r in rr]
def checked_payload(root,e):
 import torch
 proof={}
 for key,hkey in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:
  p=root/e[key];assert sha(p)==e[hkey],p;proof[key]=str(p);proof[hkey]=e[hkey]
 c=torch.load(root/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read'];return c,proof
