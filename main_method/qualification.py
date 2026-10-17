"""Recompute saved evidence only: code cleanup and acceleration equivalence."""
from pathlib import Path
import argparse
import ast
import numpy as np
from .runtime import ROOT,BASE,read,put,sha

def cpu():
    import torch
    from scripts import stvg_temporal_marginal_train_fit256_v1 as fit
    from scripts.stvg_temporal_marginal_decode_util256_v1 import decode
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_pipeline_speed_v1 import compare_tree
    old=ROOT/'artifacts/stvg_pairwise_caption256_v1'
    rows=read(old/'ROSTER_LOCK.json')['records']
    jobs={x['case_id']:x for x in read(old/'INPUT_BARRIER.json')['jobs']}
    fits={x['case_id']:x for x in read(old/'QWEN/FIT_BARRIER.json')['files']}
    records=[]
    for ds in ('hc2','vidstg'):
        r=next(x for x in rows if x['dataset']==ds)
        cid=r['probe_alias'];c,_=load_case(r);j=read(ROOT/jobs[cid]['path'])
        ref=torch.load(ROOT/fits[cid]['path'],map_location='cpu',weights_only=False)['fit']
        off=fit.native.offsets(c,c['temporal_hidden'])
        new=fit.fit(off,j['selection']['hypotheses'],ref['rewards'])
        compare_tree(new,ref)
        assert decode(c['records'],new['final_logits'])[0]==decode(c['records'],ref['final_logits'])[0]
        records.append(dict(dataset=ds,status='exact',optimizer_path_steps=len(new['path']),
            merged_decoder_exact=True,GT_read=False))
    for p in (ROOT/'main_method').glob('*.py'):ast.parse(p.read_text())
    put(BASE/'CPU_EQUIVALENCE_COMPLETION.json',dict(status='pass',count=2,records=records,
        actual_new_model_calls=0,new_GT_scoring=0,algorithm_changed=False,GT_read=False))

def processor():
    import hashlib
    import torch
    from transformers import AutoProcessor
    from .teacher import direct_batch
    from scripts.stvg_qwen35_geometry32_v1 import CHECKPOINT
    q=ROOT/'artifacts/stvg_qwen35_geometry32_v1'
    jobs=read(q/'INPUT_JOB_LOCK.json')['records']
    p=AutoProcessor.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False)
    records=[]
    for ds in ('hc2','vidstg'):
        for j in [x for x in jobs if x['dataset']==ds][:2]:
            im=np.load(ROOT/j['image']['path'],allow_pickle=False)
            ctx=np.load(ROOT/j['context']['path'],allow_pickle=False)
            b,a=direct_batch(p,j,im,ctx)
            prior=read(q/'formal_responses'/(j['job_id']+'.json'))['input_audit']
            assert a['input_tensor_sha256']==prior['input_tensor_sha256']
            records.append(dict(dataset=ds,status='exact',all_actual_input_tensors_exact=True))
    assert not torch.cuda.is_initialized()
    put(BASE/'PROCESSOR_EQUIVALENCE_COMPLETION.json',dict(status='pass',count=4,records=records,new_model_calls=0,GT_read=False))

def gpu_source():
    import collections,gc
    import sys
    from .runtime import interpreter
    assert Path(sys.prefix).resolve()==Path(interpreter('gpu_equivalence')).parent.parent.resolve(), 'Source qualification requires the original Student environment'
    import torch
    from . import engine
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_pipeline_speed_v1 import compare_tree,full_prefix_readout
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from scripts.audit_stvg_spatial_lite_fit002 import fit_audit
    from scripts.run_decota_paper_main_v1 import gpu
    lease=gpu();q=ROOT/'artifacts/stvg_qwen35_geometry32_v1'
    rr=read(q/'ROSTER_LOCK.json')['records']
    refs={r['case_id']:r['anchors'] for r in read(q/'arms/QWEN_OPD/trials/G32/REFERENCE_LOCK.json')['records']}
    old={r['case_id']:r for r in read(q/'arms/QWEN_OPD/trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']}
    records=[]
    for ds in ('hc2','vidstg'):
        r=next(x for x in rr if x['dataset']==ds and refs[x['probe_alias']])
        c,_=load_case(r);m=engine.model_for(ds)
        fresh,base,*rest=engine.capture(m,r,collections.OrderedDict())
        assert fresh['pixel_sha256']==c['pixel_sha256']
        compare_tree(fresh['zero'],c['zero'])
        compare_tree(fresh['temporal_hidden'],c['temporal_hidden'])
        a=[dict(position=x['position'],box=x['box_cxcywh'],weight=x['weight']) for x in refs[r['probe_alias']]]
        result=engine.spatial(base,c,a,stable_fit_key(ds,r['query_key_sha256']))
        d=torch.load(ROOT/old[r['probe_alias']]['path'],map_location='cpu',weights_only=False)
        ref=d['fits']['M2']
        # The saved fit carries one legacy timing-only field. The current
        # fitter also exposes solver_settings as a redundant projection of
        # settings; the historical fit predates that field. Bind the projection
        # exactly, then compare every remaining fitter output and nested path.
        ref_fit={k:v for k,v in ref.items() if k!='instance_study_fit_seconds'}
        if 'solver_settings' not in ref_fit:
            ref_fit['solver_settings']={k:ref_fit['settings'][k] for k in ('mu','cg_max_iter','cg_rtol')}
        compare_tree(result,ref_fit,f'{ds}.fits.M2')
        fit_audit(result,dict(anchors=a),'lite_fullstep')
        values=full_prefix_readout(base,list(map(int,c['frame_ids'])),c['records'],result['final_state'])
        compare_tree(values,d['readouts']['M2'],f'{ds}.readouts.M2')
        records.append(dict(dataset=ds,status='pass',Source_exact=True,gradient_Adam_response_acceptance_full_path_exact=True,actual_complete_boxes_exact=True,actual_complete_readout_exact=True,new_Spatial_fits=1,GT_read=False))
        del m,base,rest,result,d;gc.collect();torch.cuda.empty_cache()
    put(BASE/'GPU_EQUIVALENCE_COMPLETION.json',dict(status='pass',count=2,records=records,new_Source_captures=2,new_Spatial_fits=2,new_Teacher_calls=0,GT_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cpu','processor','gpu_source'],default='cpu',nargs='?');a=p.parse_args();globals()[a.stage]()
