"""Finite task phases after development, not an automation or indefinite loop."""
from pathlib import Path
import sys,subprocess,os,time,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final';PY=R/'.conda/tubedetr/bin/python'
def run(name,args,timeout=1200):
 with(D/(name+'.log')).open('ab')as log:
  x=subprocess.run([str(PY),'-u','-B']+args,cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
 assert x.returncode==0,(name,x.returncode)
def main():
 assert read(D/'Live/GLOBAL_PREDICTION_BARRIER.json')['count']==128
 run('score_development',['scripts/score_stvg_referdino_finalize_v1.py','development']);run('audit_Live',['scripts/audit_stvg_referdino_finalize_v1.py','Live'])
 sel=read(D/'SELECTION_LOCK.json')
 if sel['tau']!=.05:
  for i in range(8):run(f'Fixed{i}',['scripts/run_stvg_referdino_finalize_v1.py','measure','--arm','Fixed','--chunk',str(i)],900)
  run('score_fixed',['scripts/score_stvg_referdino_finalize_v1.py','fixed']);run('audit_Fixed',['scripts/audit_stvg_referdino_finalize_v1.py','Fixed'])
 else:put(P/'SELECTED_FIXED_CACHE_REUSE.json',dict(status='exact .05 controls reused',new_model_calls=0,old_results='results/stvg_error_state/2026-10-19/referdino_dualscale_hc128/RESULTS_PUBLIC.json'))
 pins={str(f.relative_to(R)):sha(f)for f in (R/'methods/stvg_referdino_opd_v1').glob('*')if f.is_file()};pins.update({str(f.relative_to(R)):sha(f)for f in [D/'PLAN_LOCK.json',D/'SELECTION_LOCK.json',R/'scripts/run_stvg_referdino_final_formal_v1.py']});
 for f in ['checkpoints/TASTVG_VidSTG.pth','checkpoints/referdino/ryt_mevis_swinb.pth','checkpoints/groundingdino-b/groundingdino_swinb_cogcoor.pth','.cache/grounding_dino_http_v1/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/model.safetensors']:pins[f]=sha(R/f)
 put(D/'FORMAL_RUNTIME_LOCK.json',dict(status='frozen_before_formal_predict_GT',pins=pins,plan_sha256=sha(D/'PLAN_LOCK.json')))
 for i in range(4):run(f'formal{i}',['scripts/run_stvg_referdino_final_formal_v1.py',str(i)],900)
 run('score_formal',['scripts/score_stvg_referdino_finalize_v1.py','formal']);run('audit_formal',['scripts/audit_stvg_referdino_finalize_v1.py','formal'])
 put(D/'SCIENTIFIC_PHASES_COMPLETE.json',dict(status='all_temperature_selected_Fixed_formal_scored_audited',publication_docs_pending=True,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:put(D/f'CLOSE_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,error=str(e),partial_preserved=True));raise
