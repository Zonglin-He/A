"""Release only prelocked fixed prediction scoring after both audit gates."""
from common69 import *
H=ROOT/'research/round69';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pred=ROOT/p['prediction_root'];seal=read(pred/'SEALED.json');cpu=read(H/'independent_audit_attempt001/AUDIT_RESULT.json');gpu=read(H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json');assert seal['n']==2304 and seal['arm_names']==ARMS and cpu['n']==2304 and gpu['audited_inputs']==72
for cert in [cpu,gpu]:assert cert['state'].startswith('PASS_') and cert['prediction_seal_sha256']==sha(pred/'SEALED.json')
pin(read(H/'experiment/DERIVED_SUMMARIES_PROGRAM_LOCK.json'))
s=read(H/'STATE.json');assert s['execution_open'] and s['manual_authorization'] and not s['CPU_scoring_authorized'];s.update(state='R69_ALL_FIXED_PREDICTIONS_AND_TWO_AUDIT_GATES_PASSED_SCORING_RELEASED',CPU_scoring_authorized=True);dump(H/'STATE.json',s)
dump(H/'experiment/SCORING_RELEASE_RECEIPT.json',{'state':s['state'],'n':2304,'prediction_seal_sha256':sha(pred/'SEALED.json'),'CPU_audit_sha256':sha(H/'independent_audit_attempt001/AUDIT_RESULT.json'),'GPU_audit_sha256':sha(H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json'),'methods_or_strengths_changed':False,'FINAL_prediction_or_scoring':False});print({'state':s['state']})
