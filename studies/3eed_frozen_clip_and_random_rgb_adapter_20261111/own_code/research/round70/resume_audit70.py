"""Finite authorized stage chain; stops at first failure, no retries/search."""
from common70 import *
import subprocess,time,os
PYTHON='crossplatform/env/bin/python';LOG=H/'pipeline_logs';LOG.mkdir(exist_ok=True)
def run(script,args=[],resource='cpu'):
 cmd=[PYTHON,'tools/manual_r70_guard.py','--resource',resource,'--timeout-seconds','3600','--',PYTHON,'research/round70/'+script]+args;dump(H/'PIPELINE_STATE.json',{'stage':script,'args':args,'resource':resource});f=(LOG/(script+('_engineering' if '--engineering' in args else '')+'.log')).open('w');c=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);f.close();assert c.returncode==0,(script,c.returncode)
def stage(s):subprocess.run([PYTHON,'research/round70/stage70.py',s],cwd=ROOT,check=True)
try:
 assert (ROOT/read(H/'experiment/DESIGN_LOCK.json')['A_root']/'SEALED.json').exists() and (ROOT/read(H/'experiment/DESIGN_LOCK.json')['B_root']/'SEALED.json').exists()
 assert (ROOT/read(H/'experiment/DESIGN_LOCK.json')['prediction_root']/'SEALED.json').exists()
 run('audit70.py');run('audit_gpu70.py',resource='gpu')
 p=read(H/'experiment/DESIGN_LOCK.json');q={**p,'arm_names':ARMS,'reference_arms':['source','A_subject_joint','B_NW','R59'],'independent_audit':'research/round70/independent_audit_attempt001/AUDIT_RESULT.json','score_root':'research/round70/score_dev336_attempt001'};dump(H/'experiment/SCORE_PLAN_LOCK.json',q);s=read(H/'STATE.json');s['CPU_scoring_authorized']=True;s['all_prediction_and_independent_audit_seals_present']=True;dump(H/'STATE.json',s);run('score70.py',['--plan',str(H/'experiment/SCORE_PLAN_LOCK.json')]);run('audit_scalar70.py');dump(H/'PIPELINE_STATE.json',{'stage':'ALL_SCIENCE_COMPLETED','publication_pending':True});print('R70_SCIENCE_COMPLETE',flush=True)
except BaseException as e:dump(H/'PIPELINE_FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_automatic_retry_or_new_method':True});raise
