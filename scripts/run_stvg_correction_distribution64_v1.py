"""Finite CPU-only hypothesis/readout experiment, seal then score separately."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,save,load
D=ROOT/'artifacts/stvg_correction_distribution64_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/correction_distribution64'
PRE=ROOT/'artifacts/stvg_dino_supported_source256_v1'
PAUSED=ROOT/'artifacts/stvg_dense_e_opd64_v1'
ARMS=['Source_MBR','E_MBR','Correction_mean','Correction_MBR']


def status(stage,done,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=64,pid=os.getpid(),timestamp=time.time(),**kw),True)


def prepare():
    from vg_tta.stvg_correction_distribution64_v1 import contract
    assert not(D/'PLAN_LOCK.json').exists()and read(PAUSED/'PAUSE_RECEIPT.json')['status']=='paused_by_user'
    records=read(PAUSED/'PLAN_LOCK.json')['records'];assert len(records)==64
    for ds in ['hc2','vidstg']:assert len([r for r in records if r['dataset']==ds])==32
    own=['vg_tta/stvg_correction_distribution64_v1.py','scripts/run_stvg_correction_distribution64_v1.py','protocols/stvg_correction_distribution64_v1.md']
    pins={**read(PRE/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in own}}
    put(D/'PLAN_LOCK.json',dict(status='locked_before_new_readout_and_GT',records=records,count=64,arms=ARMS,
        exact_predecessor_roster_plan_sha256=sha(PAUSED/'PLAN_LOCK.json'),selection=read(PAUSED/'PLAN_LOCK.json')['selection'],
        working_weights='Gibbs over joint conditional assignment energy / existing source cutoff; exact delta aliases mass-summed',
        mean='apply weighted mean correction in cx,cy,logw,logh',
        utility='physical-time-cell mean box IoU on common predicted MBR support, original decoder grid',
        temperature_multiplier=1.,beta=1/16,no_GT_read_for_selection_or_weights=True,
        historical_development_exposure=True,independent_confirmation=False,new_model_calls=0,new_DINO_calls=0,new_Student_updates=0,
        source_calibration_sha256=sha(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json'),
        CPU_cap_seconds=1800,created_epoch=time.time(),deadline_epoch=time.time()+7200))
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,plan_sha256=sha(D/'PLAN_LOCK.json'),registries=read(PRE/'RUNTIME_LOCK.json')['registries']))
    put(D/'CPU_CONTRACT_COMPLETION.json',contract());status('prepared',0,GT_read=False)
    print(json.dumps(dict(status='prepared',count=64,new_model_calls=0)))


def verify():
    from main_method.runtime import resource_guard
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert time.time()<p['deadline_epoch']and sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    for f,h in {**l['pins'],**l['registries']}.items():assert sha(ROOT/f)==h,f
    resource_guard();return p


def predict():
    import torch,numpy as np
    from vg_tta.stvg_correction_distribution64_v1 import decide
    from vg_tta.stvg_error_state_correction_v1 import encode
    p=verify();cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');files=[];start=time.perf_counter()
    for i,r in enumerate(p['records']):
        verify();assert time.perf_counter()-start<p['CPU_cap_seconds'];f=ROOT/r['prior_prediction'];assert sha(f)==r['prediction_sha256'];z=load(f)
        base=z['outputs']['Source_no_correction'];frames=z['frames'];pools=[encode(v['boxes'])-encode(base)[v['position']]for v in frames]
        setting=cal['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2'];tick=time.perf_counter()
        decision=decide(base,pools,[v['scores']for v in frames],setting['scale'],setting['truncation'],z['frame_ids'],z['interval'])
        h=decision['distribution'];mincost=min(v['energy']for v in h['assignments']);assert abs(mincost-z['estimate']['objective'])<1e-8
        assert np.min(np.max(abs(h['deltas']-z['estimate']['delta']),axis=1))<1e-10
        outputs=dict(Source_MBR=base,E_MBR=z['outputs']['E_DINO_support'],Correction_mean=decision['mean_output'],Correction_MBR=decision['MBR_output'])
        files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=r['query_ordinal'],
            frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=z['planned_positions'],frames=frames,setting=setting,decision=decision,outputs=outputs,
            prior_prediction_sha256=r['prediction_sha256'],GT_read=False,compute=dict(CPU_readout_seconds=time.perf_counter()-tick,new_model_calls=0,new_DINO_calls=0,new_Student_updates=0))))
        status('sealing_frozen_output_readouts',i+1,GT_read=False);print(r['alias'],i+1,'of64',flush=True)
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=files,GT_read=False,all_outputs_before_GT=True,CPU_wall_seconds=time.perf_counter()-start,new_model_calls=0,new_DINO_calls=0,new_Student_updates=0))
    assert not torch.cuda.is_initialized()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict']);a=p.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),partial_preserved=True));raise
