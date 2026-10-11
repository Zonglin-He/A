"""Exact reconstruction against the already sealed real P1 DINO readout. No GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha,load
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1';OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'

def run():
    from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate
    activate()
    from scripts.stvg_opd_paper_later_common_v1 import record_path
    from scripts.decota_paper_common_v1 import load_npz
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_paper_controls_v1 import dino_refine
    write(BASE/'DINO_RECONSTRUCTION_RUNTIME.json',dict(status='locked_before_CPU_reconstruction',GT_read=False,pins={f:sha(ROOT/f) for f in [
      'scripts/verify_stvg_opd_mechanism_DINO_v1.py','vg_tta/decota_paper_controls_v1.py']},P1_stateless_global_sha256=sha(PAPER/'table1_stateless/GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
    count=coordinates=0;records=[]
    for ds,job in [('hc2','t1_ours_hc2'),('vidstg','t1_ours_vid')]:
        stage='P2_'+ds+'_cross_clean';b=read(OLD/'stages'/stage/'PREDICTION_BARRIER.json');assert b['status']=='sealed' and not b['GT_read']
        barrier=read(PAPER/'table1_stateless'/job/'PREDICTION_BARRIER.json')
        for key,binding in b['logical_records'].items():
            condition,order,arm,index=key.split('/')
            if arm!='on_policy':continue
            p,_=record_path(b,condition,order,arm,int(index));z=load(p);ip=OLD/z['input']['path'];assert sha(ip)==z['input']['sha256'];inp=load(ip)
            q=binding['query_ordinal'];oldpath=PAPER/'table1_stateless'/job/f'{q:05}.npz';a,md,rc=load_npz(oldpath)
            assert sha(oldpath)==barrier['files'][str(oldpath.relative_to(PAPER))]
            assert md['frame_ids']==inp['frame_ids'] and md['interval']==inp['interval']
            assert np.array_equal(a['Source'],inp['native_boxes'].numpy())
            refined,meta=dino_refine(inp['native_boxes'].numpy(),inp['frame_ids'],unpack_expert(inp['expert']))
            assert np.array_equal(refined,a['DINO_Refine']) and meta==md['control']
            count+=1;coordinates+=refined.size;records.append(dict(dataset=ds,query_ordinal=q,order=order,input_sha256=sha(ip),original_P1_stateless_sha256=sha(oldpath),bitwise=True))
    assert count==512
    write(BASE/'DINO_RECONSTRUCTION_READBACK.json',dict(status='pass',scope='all clean P2 inputs versus actual sealed P1 Source and DINO trajectories',
        matched_clean_arrivals=count,DINO_coordinates=coordinates,source_coordinates=coordinates,new_model_calls=0,new_DINO_calls=0,GT_read=False,
        records=records,runtime_sha256=sha(BASE/'DINO_RECONSTRUCTION_RUNTIME.json'),time=time.time()))
    print('DINO_RECONSTRUCTION_PASS',count,coordinates,flush=True)

if __name__=='__main__':run()
