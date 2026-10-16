"""Original complete-model worker dispatched to locked Full/Lite recipes."""
import copy
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,SEARCH,read,verify,event_output_positions


def main():
    import torch
    from scripts import run_stvg_soft_ep_coordinate_trial003 as worker
    from scripts import stvg_soft_ep_coordinate_fit002 as dispatch
    from scripts import stvg_component_spatial_fit_v1 as full
    from scripts import stvg_spatial_lite_fit_v1 as lite
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import run_stvg_adjacent_motion_student_v1 as readout_module
    verify()
    tid=sys.argv[1];design=read(BASE/'trials'/tid/'DESIGN_LOCK.json')
    trows={r['case_id']:r for r in read(BASE/'trials'/tid/'TEMPORAL_LOCK.json')['records']}
    key_intervals={worker._stable_fit_key(r):trows[r['probe_alias']]['selected_interval'] for r in read(SEARCH/'ROSTER_LOCK.json')['records']}
    oldfit=dispatch.fit
    def fitter(base,initial,anchors,ids,key,cfg,mode,settings,qualification=False,checkpoint=None):
        interval=key_intervals[key]
        sync=lambda:torch.cuda.synchronize() if torch.cuda.is_initialized() else None
        if design['recipe']=='full':
            oldreplay=core._trainable_replay
            counts={'all_replay_values':0}
            def trainable(*a,**kw):
                rp,selected=oldreplay(*a,**kw);oldvalues=rp.values
                def values():
                    counts['all_replay_values']+=1
                    return oldvalues()
                rp.values=values
                return rp,selected
            core._trainable_replay=trainable
            try:
                sync();start=time.perf_counter()
                f=full.fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=qualification,checkpoint=checkpoint,flags={})
                sync();elapsed=time.perf_counter()-start
            finally:core._trainable_replay=oldreplay
        else:
            positions=event_output_positions(ids,interval) if design['response_scope']=='event_support' else None
            sync();start=time.perf_counter()
            f=lite.fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=qualification,checkpoint=checkpoint,recipe=design['recipe'],response_scope=design['response_scope'],event_support_positions=positions)
            sync();elapsed=time.perf_counter()-start
            counts=copy.deepcopy(f['decode_counts'])
        f['matched_efficiency']=dict(fit_wall_cuda_synchronized_seconds=elapsed,decode_counts=counts,source_capture_included=False,full_model_final_redecode_included=False,qualification_enabled=qualification)
        f['study_recipe']=design['recipe'];f['study_response_scope']=design['response_scope'];f['predicted_event_interval']=interval
        return f
    oldreadout=readout_module._readout
    def timed_readout(*a,**kw):
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        start=time.perf_counter();out=oldreadout(*a,**kw)
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        out['matched_complete_forward_wall_seconds']=time.perf_counter()-start
        return out
    try:
        readout_module._readout=timed_readout
        dispatch.fit=fitter;worker.BASE=BASE;worker.TRIALS=BASE/'trials';worker.MODES=('M2',);worker.ARM_NAMES=('F','T','S_M2','TS_M2');worker.verify_runtime=verify
        worker.main()
    finally:
        dispatch.fit=oldfit;readout_module._readout=oldreadout


if __name__=='__main__':main()
