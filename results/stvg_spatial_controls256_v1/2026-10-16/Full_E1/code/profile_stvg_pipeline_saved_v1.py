"""Read saved engineering timings only; no inference or task/GT scoring."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import SMALL
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.stvg_native_instance_common_v1 import read,write,sha
def main():
    import torch
    rows=[]
    for arm in ('A','B','C'):
        b=SMALL/'trials'/arm;bp=b/'STUDENT_GLOBAL_BARRIER.json';bar=read(bp);assert bar['status']=='complete' and bar['count']==16 and not bar['GT_read'];s=f=solver=decoder=0.;size=0
        for entry in bar['files']:
            p=ROOT/entry['path'];assert sha(p)==entry['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);assert not d['GT_read'];size+=p.stat().st_size;s+=d['source_capture_seconds'];fit=d['fits']['M2']
            if fit:f+=fit['instance_study_fit_seconds'];solver+=fit['solver_seconds_total'];decoder+=sum(fit['decoder_seconds_by_purpose'].values())
        e=SMALL/'execution'/arm/'student_worker/WORKER_EXIT.json';exit_=read(e);assert exit_['exit_code']==0
        wall=exit_['wall_seconds'];rows.append(dict(arm=arm,count=16,worker_seconds=wall,source_seconds=s,fit_seconds=f,solver_seconds_within_fit=solver,decoder_seconds_within_fit=decoder,residual_including_final_full_readout_hash_IO_and_other_seconds=wall-s-f,source_fraction=s/wall,fit_fraction=f/wall,solver_fraction=solver/wall,student_payload_bytes=size,barrier_sha256=sha(bp),worker_exit_sha256=sha(e),GT_read=False))
    write(speed.BASE/'SAVED_BASELINE_PROFILE.json',dict(status='complete',count=48,rows=rows,CPU_only=True,GT_read=False,new_model_calls=0,new_GT_reads=0,scope='old completed16 parents per arm; historical engineering measurement, not current256 speed result',timing_overlap='solver/decoder are inside fit; residual is not all IO'))
    print('SAVED_BASELINE_PROFILE_COMPLETE_NO_MODEL_NO_GT',[(r['arm'],r['worker_seconds']) for r in rows])
if __name__=='__main__':main()
