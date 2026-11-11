"""Finite fixed chain, no retries/search, respects the single-GPU guard."""
from common71 import *
import subprocess,time
P='crossplatform/env/bin/python';logs=H/'pipeline_logs';logs.mkdir(exist_ok=True)
def run(script,resource='cpu'):
 dump(H/'PIPELINE_STATE.json',{'stage':script,'resource':resource});f=(logs/(script+'.log')).open('w');r=subprocess.run([P,'tools/manual_r71_guard.py','--resource',resource,'--timeout-seconds','3600','--',P,'research/round71/'+script],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);f.close();assert r.returncode==0,(script,r.returncode)
def stage(name):subprocess.run([P,'research/round71/stage71.py',name],cwd=ROOT,check=True)
try:
 while (H/'RUNNING_GPU.json').exists():time.sleep(5)
 p=read(H/'experiment/DESIGN_LOCK.json');assert (ROOT/p['clip_root']/'SEALED.json').exists() and (ROOT/p['prepared_root']/'SEALED.json').exists(),'Missing upstream seal; inspect receipts, never rerun automatically'
 stage('predict');run('predict71.py');stage('audit');run('audit71.py');run('audit_gpu71.py','gpu');stage('score');run('score71.py');run('audit_scalar71.py');run('analyze71.py');run('audit_derived71.py');run('write_report71.py');dump(H/'PIPELINE_STATE.json',{'stage':'SCIENCE_COMPLETED','publication_pending':True});print('R71_FIXED_EVALUATION_SCIENCE_COMPLETED',flush=True)
except BaseException as e:dump(H/'PIPELINE_FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_automatic_retry_or_parameter_change':True});raise
