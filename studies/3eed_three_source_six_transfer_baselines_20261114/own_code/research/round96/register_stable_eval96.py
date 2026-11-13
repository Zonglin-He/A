"""Lock fixed FP32 kernels before any target scoring; preserve unstable evaluations unchanged."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round96';E=H/'experiment';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert rd(E/'FP32_EVAL_PREFLIGHT.json')['state'].startswith('PASS_')
config='''\ndef eval_configure(plan):
 configure(plan)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True)
'''
for old,new in [('run_matrix96_attempt002.py','run_matrix96_attempt003.py'),('audit_native_matrix96.py','audit_native_matrix96_attempt003.py')]:
 s=(H/old).read_text().replace('from pathlib import Path',"import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nfrom pathlib import Path").replace('MATRIX_REPAIR_LOCK_attempt002.json','MATRIX_REPAIR_LOCK_attempt003.json').replace('MATRIX_REPAIR_RELEASE_attempt002.json','MATRIX_REPAIR_RELEASE_attempt003.json').replace("B/'frozen_matrix'","B/'frozen_matrix_attempt003'").replace("B/'frozen_matrix/MATRIX_SEALED.json'","B/'frozen_matrix_attempt003/MATRIX_SEALED.json'").replace("configure({'seed':0,'torch_num_threads':4})","eval_configure({'seed':0,'torch_num_threads':4})")
 pos=s.index('def main():');s=s[:pos]+config+s[pos:];p=H/new;assert not p.exists();p.write_text(s)
for old,new in [('score_matrix96.py','score_matrix96_attempt003.py'),('audit_matrix_summary96.py','audit_matrix_summary96_attempt003.py')]:
 s=(H/old).read_text().replace('frozen_matrix/MATRIX_SEALED.json','frozen_matrix_attempt003/MATRIX_SEALED.json').replace('frozen_scores_attempt001','frozen_scores_attempt003');p=H/new;assert not p.exists();p.write_text(s)
p=rd(E/'MATRIX_REPAIR_LOCK_attempt002.json');p['code_sha256']={'research/round96/'+n:sha(H/n) for n in ['run_matrix96_attempt003.py','audit_native_matrix96_attempt003.py','score_matrix96_attempt003.py','audit_matrix_summary96_attempt003.py']};p['code_sha256']['crossplatform/round96/source_validation.py']=sha(R/'crossplatform/round96/source_validation.py');p.update(evaluation_only_backend_repair=True,CUDNN_benchmark=False,TF32_allowed=False,torch_deterministic_algorithms=True,CUBLAS_WORKSPACE_CONFIG=':4096:8',training_weights_changed=False,task_GT_numeric_scores_before_backend_choice=0,unstable_prediction_directory_preserved='crossplatform/round96/frozen_matrix',stable_prediction_directory='crossplatform/round96/frozen_matrix_attempt003',repeated_forward_preflight_sha256=sha(E/'FP32_EVAL_PREFLIGHT.json'))
lock=E/'MATRIX_REPAIR_LOCK_attempt003.json';assert not lock.exists();lock.write_text(json.dumps(p,indent=2)+'\n');(E/'MATRIX_REPAIR_RELEASE_attempt003.json').write_text(json.dumps({'authorized':True,'plan_sha256':sha(lock)},indent=2)+'\n')
for n in ['export96.py','audit_export96.py']:
 f=H/n;f.write_text(f.read_text().replace('frozen_scores_attempt001','frozen_scores_attempt003').replace('frozen_matrix/MATRIX_SEALED.json','frozen_matrix_attempt003/MATRIX_SEALED.json').replace('MATRIX_REPAIR_LOCK_attempt002.json','MATRIX_REPAIR_LOCK_attempt003.json'))
print({'registered_backend_repair':sha(lock),'source_training_repeated':False,'all_previous_evaluation_and_audit_failures_preserved':True})
