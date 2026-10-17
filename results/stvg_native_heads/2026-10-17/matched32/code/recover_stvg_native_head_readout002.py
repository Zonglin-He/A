"""CPU only: native cxcywh -> evaluator xyxy, immutable raw/model outputs kept."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_native_head_adapter32_v1 as old
BASE=old.BASE/'readout002';read,write,sha=old.read,old.write,old.sha
def verify():
    old.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME002.json')['pins'].items():assert sha(ROOT/p)==h,p
def init():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import save_pt
    old.verify();assert not BASE.exists();BASE.mkdir()
    for n in ['ROSTER_LOCK.json','METHOD_LOCK.json','INPUT_JOB_LOCK.json','QUALIFICATION_COMPLETION.json','CPU_CONTRACT_COMPLETION.json']:(BASE/n).write_bytes((old.BASE/n).read_bytes())
    files=[];checks=0
    for f in read(old.BASE/'GLOBAL_PREDICTION_BARRIER.json')['files']:
        assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);fit=d['fit']
        if fit is not None:
            raw=fit['final_cached_decoder_boxes'].numpy().astype(float);assert np.array_equal(raw,np.asarray(d['arms']['SPATIAL_HEAD']['boxes']))
            # Same native box convention used by all previous Source/Student producers.
            xyxy=np.concatenate((raw[:,:2]-raw[:,2:]/2,raw[:,:2]+raw[:,2:]/2),axis=1)
            for arm in ['SPATIAL_HEAD','BOTH_HEADS','NEW_S_FIXED_CLOCK']:d['arms'][arm]['boxes']=xyxy.copy();checks+=len(raw)
            non=[i for i in range(len(raw))if i not in d['support']];source=np.asarray(d['arms']['Frozen']['boxes']);d['diagnostics']['nonanchor_changed_frames']=sum(not np.array_equal(xyxy[i],source[i])for i in non)
        d['raw_model_output_sha256']=f['sha256'];d['CPU_readout_conversion']='native cxcywh to evaluator normalized xyxy, no clamp or model rerun'
        r=save_pt(BASE/'predictions'/(d['case_id']+'.pt'),d);r['case_id']=d['case_id'];files.append(r)
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));write(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,parent_actual_forward_root_sha256=sha(old.BASE/'ROOT_READBACK.json'),native_format_conversion_checks=checks,actual_model_outputs_unchanged=True,new_models=0,new_fits=0,GT_read=False))
    paths=[Path(__file__),old.BASE/'EXECUTION_RUNTIME.json',old.BASE/'GLOBAL_PREDICTION_BARRIER.json',BASE/'GLOBAL_PREDICTION_BARRIER.json'];write(BASE/'EXECUTION_RUNTIME002.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False))
    write(old.BASE/'recovery/readout_format002/RECOVERY.json',dict(status='explained_engineering_failure',old_score_failure='native cxcywh accidentally sent to xyxy evaluator',old_outputs_preserved=True,derived_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),new_models=0,new_Teacher_calls=0,new_fits=0,no_GT_rule_change=True,GT_read=False))
def run(stage):
    verify()
    if stage=='score':
        ns=dict(old.__dict__,BASE=BASE,verify=verify);exec(compile(inspect.getsource(old.score),__file__,'exec'),ns);return ns['score']()
    from scripts import report_stvg_native_head_adapter32_v1 as report
    report.BASE=BASE;return report.report()
if __name__=='__main__':
    if sys.argv[1]=='init':init()
    else:run(sys.argv[1])
