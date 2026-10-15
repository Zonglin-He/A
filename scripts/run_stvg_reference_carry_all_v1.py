"""Finite, serial driver for the six prelocked LN-carry streams; no GT."""
from pathlib import Path
import os,sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_recovery_common_v1 import BASE,read,write,sha,verify_runtime

def main():
 verify_runtime();start=time.perf_counter();entries=read(BASE/'ORDERS.json')['records'];assert len(entries)==6
 files=[];streams=[];steps=fits=readouts=0
 for entry in entries:
  ds,order=entry['dataset'],entry['order'];path=BASE/'student'/ds/order/'STREAM_BARRIER.json'
  cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_stvg_reference_carry_v1.py','--dataset',ds,'--order',order]
  subprocess.run(cmd,cwd=ROOT,check=True)
  d=read(path);assert d['status']=='complete' and d['count']==16 and d['case_ids']==entry['case_ids'] and d['GT_read'] is False
  for item in d['files']:assert sha(ROOT/item['path'])==item['sha256']
  files.extend(d['files']);steps+=d['optimizer_steps'];fits+=d['actual_fit_count'];readouts+=d['actual_full_readout_calls']
  streams.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),dataset=ds,order=order))
  write(BASE/'STATUS.json',dict(status='running',done=len(files),total=96,streams_done=len(streams),accepted_optimizer_steps=steps,GT_read=False),mutable=True)
 assert len(files)==96 and len({x['path'] for x in files})==96 and readouts==192 and fits==96
 write(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=96,independent_source_count=32,streams=streams,files=files,actual_fit_invocations=fits,accepted_optimizer_steps=steps,actual_full_readout_calls=readouts,wall_seconds=time.perf_counter()-start,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False))
 write(BASE/'STATUS.json',dict(status='pending_independent_root',done=96,total=96,streams_done=6,accepted_optimizer_steps=steps,GT_read=False),mutable=True)
 print('six carry streams sealed',len(files),steps,flush=True)
if __name__=='__main__':main()
