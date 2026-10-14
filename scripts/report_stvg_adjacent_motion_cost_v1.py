"""CPU readback of actual measured stage, call, source and fit costs."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';O=ROOT/'results/stvg_adjacent_motion_opd/2026-10-14'
def read(p):return json.loads(p.read_text())
def main():
 rows=[];seal=read(B/'METHOD_STUDENT_GLOBAL_BARRIER.json');assert seal['status']=='complete'
 for f in seal['files']:
  p=ROOT/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];alias=('HC'if cid.startswith('hc2')else'V')+f'{int(cid.split("_")[-1])+1:02d}'
  fits=[dict(arm=k,elapsed_seconds=v['elapsed_seconds'],optimizer_steps=v['optimizer_steps'],actual_final_readouts=v['final_redecode_calls']) for k,v in d['reference_receipts'].items() if 'elapsed_seconds'in v]
  rows.append(dict(case_id=alias,source_capture_seconds=d['source_cold_elapsed_seconds'],actual_fits=fits,source_file_sha256=f['sha256']))
 walls=[]
 for stage in ['TEACHER','STUDENT']:
  l=read(B/(stage+'_LAUNCH.json'));e=read(B/(stage+'_WORKER_EXIT.json'));assert e['exit_code']==0;walls.append(dict(stage=stage,wall_seconds=e['time']-l['time'],exit_code=0,includes_stage_load_and_io=True))
 failed=read(B/'recovery/observation_dispatch001/CANDIDATE_INPUT_BARRIER.json')
 out=dict(status='pass',scope='actual timers, no estimation; nested counters must not be summed',cases=rows,stage_walls=walls,teacher=read(B/'ROOT_TEACHER_READBACK.json')['cost'],successful_preparation_seconds=read(B/'CANDIDATE_INPUT_BARRIER.json')['elapsed_cpu_seconds'],preserved_failed_preparation_seconds=failed['elapsed_cpu_seconds'],fit_count=sum(len(r['actual_fits'])for r in rows),optimizer_steps=sum(v['optimizer_steps']for r in rows for v in r['actual_fits']),source_capture_count=len(rows),model_loaded=False,new_GPU_calls=0)
 assert out['fit_count']==9 and out['optimizer_steps']==45 and len(rows)==8
 O.mkdir(parents=True,exist_ok=True);(O/'actual_cost_all_paths.json').write_text(json.dumps(out,indent=2)+'\n');print('cost pass',out['fit_count'],out['optimizer_steps'],out['stage_walls'])
if __name__=='__main__':main()
