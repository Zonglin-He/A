"""Read actual saved timing counters on CPU, with repaired-attempt overhead separated."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1';O=ROOT/'results/stvg_motion_guided_instance_opd/2026-10-14'
def read(p):return json.loads(p.read_text())
def main():
 rows=[]
 for phase,prefix in [('method','METHOD'),('reference_control','REFERENCE_CONTROL')]:
  seal=read(B/(prefix+'_STUDENT_GLOBAL_BARRIER.json'))
  for f in seal['files']:
   p=ROOT/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];alias=('HC'if cid.startswith('hc2')else'V')+f'{int(cid.split("_")[-1])+1:02d}'
   rr=d['reference_receipts']if phase=='method'else d['reference_control_roles']
   fit=[dict(arm=k,elapsed_seconds=v['elapsed_seconds'],optimizer_steps=v['optimizer_steps'],actual_final_readouts=v['final_redecode_calls'])for k,v in rr.items()if 'elapsed_seconds'in v]
   rows.append(dict(case_id=alias,phase=phase,source_capture_seconds=d['source_cold_elapsed_seconds'],actual_fits=fit,source_file_sha256=f['sha256']))
 walls=[]
 for base,launch,ex in [(B,'TEACHER_LAUNCH','TEACHER_WORKER_EXIT'),(B,'METHOD_LAUNCH','METHOD_WORKER_EXIT'),(B,'METHOD_LAUNCH002','METHOD_WORKER_EXIT002'),(B,'REFERENCE_CONTROL_LAUNCH002','REFERENCE_CONTROL_WORKER_EXIT002'),(B/'display_order_layout001','TEACHER_LAUNCH','TEACHER_WORKER_EXIT')]:
  l=read(base/(launch+'.json'));e=read(base/(ex+'.json'));walls.append(dict(stage=base.name+'/'+launch,wall_seconds=e['time']-l['time'],exit_code=e['exit_code'],includes_stage_load_and_io=True))
 out=dict(status='pass',scope='actual recorded timing; per-fit includes final readout, stage walls include load and may overlap; do not sum nested counters',cases=rows,stage_attempt_walls=walls,teacher=read(B/'ROOT_TEACHER_FINAL_READBACK003.json')['actual_cost'],fit_count=sum(len(r['actual_fits'])for r in rows),optimizer_steps=sum(v['optimizer_steps']for r in rows for v in r['actual_fits']),source_capture_count=len(rows),method_partial_retry_overhead_in_attempt_wall_not_unique_fit_sum=True,model_loaded=False,new_GPU_calls=0)
 assert out['fit_count']==21 and out['optimizer_steps']==90
 (O/'actual_cost_all_paths.json').write_text(json.dumps(out,indent=2)+'\n');print('cost readback pass 21 fits 90 steps 17 source captures')
if __name__=='__main__':main()
