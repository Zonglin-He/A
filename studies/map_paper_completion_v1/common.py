"""Independent current-MAP paper completion; old evidence is read-only."""
import os,sys,json,hashlib,time,csv
from pathlib import Path
from functools import lru_cache
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/map_full_v1'),str(R/'studies/paper_review_suite_v1')]
from full import RUN,PROVIDER,read,sha,atomic,identity,check,binding
D=R/'outputs/map_paper_completion_v1';P=R/'results/map_paper_completion_v1';OLD=R/'outputs/paper_review_suite_v1';MAP=R/'outputs/map_full_now_v3'
COUNTS={'hc2':1902,'vidstg':10303}
fixed=lru_cache(maxsize=None)(read)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def records(ds):
 j=read(RUN/'plan.json')['jobs'][ds];p=PROVIDER/j['manifest'];assert sha(p)==j['manifest_sha256'];return list(map(json.loads,p.read_text().splitlines()))
def labels(ds):
 j=read(RUN/'plan.json')['jobs'][ds];p=PROVIDER/j['labels'];assert sha(p)==j['labels_sha256'];return {x['id']:x for x in map(json.loads,p.read_text().splitlines())if x['dataset']==ds}
def freeze(stage,science,files):
 check();assert not(D/f'{stage}_plan.json').exists()
 p=dict(schema='current_MAP_completion_'+stage,science=science,suite_plan_sha256=sha(D/'plan.json'),parent_plan_sha256=sha(RUN/'plan.json'),parent_seals={ds:sha(RUN/ds/'sealed.json')for ds in COUNTS},MAP_seals={ds:sha(MAP/f'TA_{ds}_sealed.json')for ds in COUNTS},code_sha256={str(f.relative_to(R)):sha(f)for f in files},GT_in_prediction=False)
 p['fingerprint']=digest(p);atomic(D/f'{stage}_plan.json',p);atomic(P/f'{stage}_protocol.json',p)
 for f in files:
  q=D/'source_snapshot'/stage/f.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
 return p
def validate(stage):
 p=read(D/f'{stage}_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})and p['suite_plan_sha256']==sha(D/'plan.json')and p['parent_plan_sha256']==sha(RUN/'plan.json')
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for ds,h in p['parent_seals'].items():assert sha(RUN/ds/'sealed.json')==h
 for ds,h in p['MAP_seals'].items():assert sha(MAP/f'TA_{ds}_sealed.json')==h
 check();return p
def load_parent(ds,row):
 import torch
 e=fixed(RUN/ds/'sealed.json')['files'][row['id']];cp=RUN/ds/e['capture_path'];pp=RUN/ds/e['path'];rp=RUN/ds/e['receipt_path']
 assert sha(cp)==e['capture_sha256']and sha(pp)==e['sha256']and sha(rp)==e['receipt_sha256']
 assert read(rp)['input_binding']==binding(row,read(RUN/'plan.json')['provider_runtime_fingerprint'])
 c=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read'];return c,dict(capture=str(cp),capture_sha256=sha(cp),prediction=str(pp),prediction_sha256=sha(pp),receipt=str(rp),receipt_sha256=sha(rp))
def current_MAP(ds,key):
 e=fixed(MAP/f'TA_{ds}_sealed.json')['files'][key];p=MAP/e['path'];assert sha(p)==e['sha256'];return p,e['sha256']
def parent_report(rows,arms,pairs,folder):
 from studies.paper_followup_v1.common import report
 return report(rows,arms,pairs,folder)
def signed_tails(rows,pairs,folder):
 with(folder/'all_signed_severe_timegroups.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','timegroup','comparison','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_union'])
  for i,r in enumerate(rows,1):
   for a,b in pairs:
    dt=100*(r['values'][a]['t']-r['values'][b]['t']);dv=100*(r['values'][a]['v']-r['values'][b]['v']);w.writerow([f'Q{i:05}',r['duration_group'],a+'-'+b,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5,dt>5 or dv>5])
def owned_exit(w):
 try:s=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return s[0]=='Z'or int(s[19])!=w['startticks']
 except FileNotFoundError:return True
