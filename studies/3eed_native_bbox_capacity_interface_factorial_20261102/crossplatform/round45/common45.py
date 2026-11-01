"""R45 missing7B-bbox cell; reuse frozen loading/durability, no task reader."""
import sys,json,os,time,hashlib,datetime,signal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round42'),str(ROOT/'crossplatform/round40'),str(ROOT/'crossplatform/round39'),str(ROOT/'crossplatform/round37')]
from common42 import atomic,atomic_npz,capture_decoded_raw,memory_receipt,start_wall_budget,select_preflight
from common39 import rows,prepared,arrays,read,sha,verify,load_npz,array_sha,messages_bbox,parse_bbox,bbox_readout
from common40 import ARMS as CACHED_ARMS,iou_assignment,iou_readout
BASE=ROOT/'crossplatform/round45';EXP=ROOT/'research/round45/experiment';OUT=BASE/'bboxA208_attempt001'
P3=ROOT/'crossplatform/round39/inference208_attempt002';MATCHED=ROOT/'crossplatform/round40/predictions208_attempt001';POINT7=ROOT/'crossplatform/round42/pointA208_attempt001'
MODEL=ROOT/'crossplatform/assets/Qwen2.5-VL-7B-Instruct'
POINT_ARMS=['7B_point_compatible_fullmap','7B_point_compatible_oldsubset']
NEW_ARMS=['7B_bbox_center_fullmap','7B_bbox_center_oldsubset','7B_bbox_rectIoU_fullmap','7B_bbox_rectIoU_oldsubset']
ARMS=CACHED_ARMS+POINT_ARMS+NEW_ARMS
def check_authorization():
 p=read(EXP/'PLAN_LOCK.json');i=read(EXP/'IMPLEMENTATION_LOCK.json');a=read(EXP/'EXECUTION_AUTHORIZATION.json')
 assert a['model_GPU_execution_authorized'] is True and a['plan_sha256']==sha(EXP/'PLAN_LOCK.json') and a['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')
 for f,h in i['program_sha256'].items():assert sha(ROOT/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(ROOT/f)==h,f
 c=read(EXP/'CPU_TECH_CHECKS.json');assert c['state']=='PASSED' and c['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')
 path=MODEL/'DOWNLOAD_MANIFEST_ATTEMPT002.json';assert sha(path)==a['complete_download_manifest_sha256'];d=read(path);assert d['state']=='COMPLETE_VERIFIED_NO_INFERENCE' and d['revision']==p['model_revision'];files={r['filename']:r for r in d['files']}
 for n,v in p['model_files'].items():assert files[n]['size']==v['size']==(MODEL/n).stat().st_size and sha(MODEL/n)==v['sha256']
 assert not list(MODEL.glob('*.safetensors.part'));return p,a,d
def annotate_saved(rawpath,input_binding,tokenpath,resultpath):
 raw=read(rawpath)['raw_answer_PRIVATE'];tokens=read(tokenpath)['generated_token_ids']
 result={**input_binding,'raw_answer_PRIVATE':raw,'parsed':parse_bbox(raw,input_binding['model_resized_wh']),'generated_tokens':len(tokens),'ended_with_eos':any(x in input_binding['eos_token_ids'] for x in tokens),'raw_answer_file':str(Path(rawpath).relative_to(OUT)) if Path(rawpath).is_relative_to(OUT) else str(rawpath),'raw_answer_file_sha256':sha(rawpath),'token_ids_file_sha256':sha(tokenpath),'raw_persisted_before_parse_and_derived_hashes':True}
 atomic(resultpath,result);return result
def seal_outputs(folder,extra):
 files={str(f.relative_to(folder)):sha(f) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='SEALED.json' and not f.name.endswith('.tmp')}
 atomic(folder/'SEALED.json',{'state':'R45_frozen7Bbbox_missingcell_predictions_before_task_scoring','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,**extra})
