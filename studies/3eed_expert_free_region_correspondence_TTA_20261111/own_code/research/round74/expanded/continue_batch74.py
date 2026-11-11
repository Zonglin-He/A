from batchcommon74 import *
import subprocess,time
p=read(H/'experiment/PREPARE_PLAN_LOCK.json')
while not (ROOT/p['evidence_root']/'SEALED.json').exists():time.sleep(2)
# Only bookkeeping and dispatch here. Model producers are guarded subprocesses.
for stage in ['fit','predict']:
 p=read(H/'experiment/PREPARE_PLAN_LOCK.json')
 p['programs_sha256']['research/round74/expanded/fit_batch74.py']=sha(H/'fit_batch74.py')
 p['programs_sha256']['research/round74/expanded/predict_batch74.py']=sha(H/'predict_batch74.py')
 p['inputs_sha256'][p['evidence_root']+'/SEALED.json']=sha(ROOT/p['evidence_root']/'SEALED.json')
 if stage=='predict':p['inputs_sha256'][p['fit_root']+'/SEALED.json']=sha(ROOT/p['fit_root']/'SEALED.json')
 plan=H/'experiment'/('FIT_PLAN_LOCK.json' if stage=='fit' else 'PREDICT_PLAN_LOCK_attempt002.json')
 assert not plan.exists();dump(plan,p)
 cmd=[str(ROOT/'crossplatform/env/bin/python'),str(ROOT/'tools/manual_r74_guard.py'),'--resource','cpu','--timeout-seconds','1800','--',str(ROOT/'crossplatform/env/bin/python'),str(H/('fit_batch74.py' if stage=='fit' else 'predict_batch74.py'))]
 subprocess.run(cmd,cwd=ROOT,check=True)
print({'fixed_expansion_producers_completed':2240})
