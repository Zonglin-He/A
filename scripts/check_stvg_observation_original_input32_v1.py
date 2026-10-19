"""Postseal engineering positive control, not a new method or tuning trial."""
import collections,gc,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_observation_factorial32_v1 import D,P,PRE,verify

def main():
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32
    previous=sum(read(f)['GPU_seconds']for f in (D/'allocations').glob('*.json'))
    assert previous<550
    lease=gpu();start=time.perf_counter();out=[];complete=False;model=None
    try:
        for alias in ['NH001','NV001']:
            assert previous+time.perf_counter()-start<900
            r=next(r for r in p['records']if r['alias']==alias);ds=r['dataset'];s=load(PRE/'source'/(alias+'.pt'))
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            model=model_for('vidstg'if ds=='hc2'else'hcstvg2');row=read_row(ds,r['query_ordinal'])
            frames,ids,_=frames_for(ds,row,collections.OrderedDict())
            with torch.no_grad():batch,recs,base=frozen_forward(model,frames,row)
            errs=[float((a.detach().cpu()-b).abs().max())for a,b in zip(base.zero['logits'],s['logits'])]
            boxerr=float((base.zero['boxes'].detach().cpu()-s['Source']).abs().max())
            assert all(torch.equal(a.detach().cpu(),b)for a,b in zip(base.zero['logits'],s['logits']))and boxerr==0
            out.append(dict(alias=alias,original_logit_max_errors=errs,Source_box_max_error=boxerr,exact=True))
            print(alias,'original physical-grid logits and Source boxes EXACT',flush=True)
            del frames,batch,base;gc.collect();torch.cuda.empty_cache()
        put(P/'ORIGINAL_INPUT_ENGINEERING_CHECK.json',dict(status='pass',rows=out,new_original_TA_queries=2,new_offset_forwards=4,
            postseal_post_initial_scoring=True,no_configuration_or_prediction_changed=True,no_new_GT_or_task_scoring=True))
        complete=True
    finally:
        elapsed=time.perf_counter()-start;lease.close()
        put(D/'allocations'/f'engineering_{time.time_ns()}.json',dict(GPU_seconds=elapsed,complete=complete,kind='postseal_original_input_qualification',new_original_queries=2))

if __name__=='__main__':main()
