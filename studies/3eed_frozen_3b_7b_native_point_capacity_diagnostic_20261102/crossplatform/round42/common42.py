"""R42 fixed same-family capacity comparison; import-safe, no task-label reader."""
import sys,json,os,time,hashlib,datetime,signal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round38'),str(ROOT/'crossplatform/round37')]
from common37 import rows,prepared,arrays,read,sha,verify,load_npz,direct_readout,array_sha
from common38 import parse_compatible
BASE=ROOT/'crossplatform/round42';EXP=ROOT/'research/round42/experiment'
OUT=BASE/'pointA208_attempt001'
P3=ROOT/'crossplatform/round37/inference208_attempt003'
MATCHED=ROOT/'crossplatform/round38/predictions208_attempt001'
SYN3=ROOT/'crossplatform/round37/unit_diagnostic_attempt003'
MODEL=ROOT/'crossplatform/assets/Qwen2.5-VL-7B-Instruct'
ARMS=['source','R29_direct','legacy_G','3B_point_compatible_fullmap','3B_point_compatible_oldsubset','3B_som_compatible_fullmap','7B_point_compatible_fullmap','7B_point_compatible_oldsubset']
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def atomic_npz(path,**data):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def check_authorization():
 plan=read(EXP/'PLAN_LOCK.json');impl=read(EXP/'IMPLEMENTATION_LOCK.json')
 # This file must be created by root only after review and complete download.
 auth=read(EXP/'EXECUTION_AUTHORIZATION.json')
 assert auth['model_GPU_execution_authorized'] is True
 assert auth['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 assert auth['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')
 for n,h in impl['program_sha256'].items():assert sha(ROOT/n)==h,n
 for n,h in plan['input_sha256'].items():assert sha(ROOT/n)==h,n
 check=read(EXP/'CPU_TECH_CHECKS.json');assert check['state']=='PASSED' and check['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')
 manifest=MODEL/'DOWNLOAD_MANIFEST_ATTEMPT002.json';assert sha(manifest)==auth['complete_download_manifest_sha256'];d=read(manifest)
 assert d['state']=='COMPLETE_VERIFIED_NO_INFERENCE' and d['revision']==plan['model_revision']
 files={r['filename']:r for r in d['files']}
 for n,entry in plan['model_files'].items():
  assert n in files and files[n]['size']==entry['size'] and (MODEL/n).stat().st_size==entry['size']
  assert sha(MODEL/n)==entry['sha256'],n
 assert not list(MODEL.glob('*.safetensors.part'))
 return plan,auth,d
def select_preflight(allrows,registered12):
 ids={x['id_PRIVATE'] for x in registered12};chosen=[]
 for pl in ['drone','quad']:chosen.append(next(r for r in allrows if r['id'] in ids and r['platform']==pl))
 return chosen
def capture_decoded_raw(rawpath,text):
 """The very first step after decode: no parser, array/hash or evidence read."""
 assert isinstance(text,str) and not Path(rawpath).exists()
 atomic(rawpath,{'raw_answer_PRIVATE':text})
def annotate_saved(rawpath,input_binding,tokenpath,resultpath):
 # Raw is already immutable on disk; all fallible parsing/hashing follows it.
 raw=read(rawpath)['raw_answer_PRIVATE'];tokens=read(tokenpath)['generated_token_ids']
 result={**input_binding,'raw_answer_PRIVATE':raw,'parsed':parse_compatible(raw,'point'),'generated_tokens':len(tokens),'ended_with_eos':any(x in input_binding['eos_token_ids'] for x in tokens),'raw_answer_file':str(Path(rawpath).relative_to(OUT)) if Path(rawpath).is_relative_to(OUT) else str(rawpath),'raw_answer_file_sha256':sha(rawpath),'token_ids_file_sha256':sha(tokenpath),'raw_persisted_before_parse_and_derived_hashes':True}
 atomic(resultpath,result);return result
def memory_receipt(torch=None):
 status={}
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.split(':')[0] in ['VmRSS','VmHWM']:status[line.split(':')[0]]=line.split(':',1)[1].strip()
 mem={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.split(':')[0] in ['MemAvailable','MemTotal','SwapFree','SwapTotal']:mem[line.split(':')[0]]=line.split(':',1)[1].strip()
 out={'process':status,'system_memory':mem}
 if torch is not None:
  try:out['GPU']={'allocated':int(torch.cuda.memory_allocated()),'reserved':int(torch.cuda.memory_reserved()),'peak_allocated':int(torch.cuda.max_memory_allocated()),'peak_reserved':int(torch.cuda.max_memory_reserved())}
  except Exception as e:out['GPU_resource_read_error']=type(e).__name__
 return out
def start_wall_budget(seconds):
 state=read(ROOT/'research/NIGHT_20261101/STATE.json');deadline=datetime.datetime.fromisoformat(state['deadline_utc'].replace('Z','+00:00'));left=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert state['status']=='active' and left>0
 def expired(*_):raise TimeoutError('R42_owned_wall_or_night_deadline')
 signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGINT,expired);signal.setitimer(signal.ITIMER_REAL,min(seconds,left));return time.perf_counter()
def seal_outputs(folder,extra):
 # No task labels, official evaluator, media upload, remote code, or scoring.
 records={str(f.relative_to(folder)):sha(f) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='SEALED.json' and not f.name.endswith('.tmp')}
 atomic(folder/'SEALED.json',{'state':'R42_frozen_capacity_predictions_before_task_scoring','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,**extra})
