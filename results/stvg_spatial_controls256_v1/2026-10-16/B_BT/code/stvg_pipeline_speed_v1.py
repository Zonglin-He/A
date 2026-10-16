"""Exact frozen-prefix reuse, bounded allocator reuse and immutable async saves.

Only pre-update Source views are cached. Final readout recomputes BOTH decoder
passes, every trainable LN, native action gates and every output frame.
"""
from pathlib import Path
import collections,copy,gc,hashlib,json,os,shutil,time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_pipeline_speed_v1'
from scripts.stvg_native_instance_common_v1 import read,write,sha

def copy_cpu(v):
    import torch
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:copy_cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [copy_cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(copy_cpu(x) for x in v)
    return copy.deepcopy(v)
def to_device(v,device):
    import torch
    if torch.is_tensor(v):return v.to(device)
    if isinstance(v,dict):return {k:to_device(x,device) for k,x in v.items()}
    if isinstance(v,list):return [to_device(x,device) for x in v]
    if isinstance(v,tuple):return tuple(to_device(x,device) for x in v)
    return v
def binding(case):
    return dict(case_id=case['case_id'],dataset=case['dataset'],source_case_sha256=case['_source_case_sha256'],query_key_sha256=case['query_key_sha256'],parent_sha256=case['parent_sha256'],subject_sha256=case['subject_sha256'],pixel_sha256=case['pixel_sha256'],frame_ids=list(map(int,case['frame_ids'])),checkpoint_sha256=case['source_checkpoint_sha256'],model_state_sha256=case['source_model_state_sha256'])
def compare_tree(a,b,where='root'):
    import torch,numpy as np
    if torch.is_tensor(a):assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),where
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),where
    elif isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(where,'keys')
        for k in a:
            if k in ('wall_seconds','step_seconds','solver_seconds','solver_seconds_total','decoder_seconds','decoder_seconds_by_purpose','elapsed_seconds','helper_elapsed_seconds','instance_study_fit_seconds','qualification_seconds'):continue
            compare_tree(a[k],b[k],where+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):compare_tree(x,y,where+f'[{i}]')
    else:assert a==b,(where,a,b)
def feature_save(case,base,records,native):
    import torch
    dest=BASE/'source_features'/(case['case_id']+'.pt');receipt=dest.with_suffix('.receipt.json');bind=binding(case)
    if receipt.exists():
        r=read(receipt);assert r['binding']==bind and sha(dest)==r['sha256'];return
    views=copy_cpu(base.views);zero=copy_cpu(base.zero);initial=copy_cpu(base.initial)
    payload=dict(schema='stvg_exact_preupdate_source_views_v1',binding=bind,views=views,zero=zero,initial=initial,records=copy_cpu(records),native=copy_cpu(native),GT_read=False)
    def nbytes(v):
        if torch.is_tensor(v):return v.numel()*v.element_size()
        if isinstance(v,dict):return sum(nbytes(x) for x in v.values())
        if isinstance(v,(list,tuple)):return sum(nbytes(x) for x in v)
        return 0
    size=nbytes(payload)
    # Reserve space for the later three Teacher arms and all six Student seals.
    if shutil.disk_usage(ROOT).free < 50*2**30+2*size:return
    dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();torch.save(payload,dest)
    write(receipt,dict(status='sealed',binding=bind,sha256=sha(dest),bytes=dest.stat().st_size,cache_boundary='original SpatialReplay.views: before all updated spatial LNs; no adapted outputs reused',GT_read=False))
def feature_restore(model,case):
    import torch
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    dest=BASE/'source_features'/(case['case_id']+'.pt');r=read(dest.with_suffix('.receipt.json'));assert r['binding']==binding(case) and sha(dest)==r['sha256']
    d=torch.load(dest,map_location='cpu',weights_only=False);assert d['binding']==binding(case) and not d['GT_read']
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert state_hash(model.state_dict())==case['source_model_state_sha256']==model._fixed_full_state_hash
    base=SpatialReplay(model,to_device(d['views'],next(model.parameters()).device),len(d['binding']['frame_ids']),cached=True)
    compare_tree(copy_cpu(base.initial),d['initial'],'restored_initial');compare_tree(copy_cpu(base.zero),d['zero'],'restored_complete_zero')
    native=prediction(base.zero['logits'],base.zero['boxes'],d['records'],d['binding']['frame_ids']);compare_tree(native,d['native'],'restored_native')
    return base,d['records'],native
def full_prefix_readout(base,ids,records,state):
    import torch
    from methods.decota_final_simplified_v1.objectives import prediction
    initial=base.state()
    try:
        base.restore({k:v.to(initial[k]) for k,v in state.items()})
        with torch.no_grad():values=base.reference_values(capture=False)
        return prediction(values['logits'],values['boxes'],records,list(ids))
    finally:base.restore(initial)

class Engine:
    def __init__(self,arm,enabled=True):
        from scripts import run_stvg_adjacent_motion_student_v1 as h
        self.arm=arm;self.enabled=enabled;self.original_capture=h._capture_source;self.original_readout=h._readout;self.base=None;self.rows=[];self.current=None;self.number=0
    def capture(self,rec,model,current_source,torch,collections_module,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash):
        from scripts import run_stvg_adjacent_motion_student_v1 as h
        case,catalog_data=h.catalog(rec);begin=time.perf_counter();row=dict(case_id=case['case_id'],dataset=case['dataset'],cache_hit=False,source_decode_seconds=0.,source_forward_seconds=0.,source_state_hash_seconds=0.,model_load_seconds=0.,GT_read=False)
        target=BASE/'source_features'/(case['case_id']+'.receipt.json')
        if self.enabled and target.exists():
            if current_source!=case['source']:
                if model is not None:del model;gc.collect();torch.cuda.empty_cache()
                t=time.perf_counter();model=model_for(case['source']);row['model_load_seconds']=time.perf_counter()-t;current_source=case['source']
            t=time.perf_counter();base,records,native=feature_restore(model,case);row['restore_seconds']=time.perf_counter()-t;row['cache_hit']=True;raw=raw_row(case);ids=list(map(int,case['frame_ids']));elapsed=time.perf_counter()-begin
            result=(model,current_source,case,catalog_data,raw,None,ids,records,None,base,native,case['source_model_state_sha256'],case['pixel_sha256'],elapsed)
        else:
            def timed(fn,key):
                def call(*a,**kw):
                    torch.cuda.synchronize();t=time.perf_counter();out=fn(*a,**kw);torch.cuda.synchronize();row[key]+=time.perf_counter()-t;return out
                return call
            result=self.original_capture(rec,model,current_source,torch,collections_module,timed(model_for,'model_load_seconds'),timed(frames_for,'source_decode_seconds'),raw_row,timed(frozen_forward,'source_forward_seconds'),prediction,timed(state_hash,'source_state_hash_seconds'))
            base,records,native=result[9],result[7],result[10]
            if self.enabled:
                t=time.perf_counter();feature_save(case,base,records,native);row['feature_store_seconds']=time.perf_counter()-t
        self.base=result[9];self.current=row;row['capture_wall_seconds']=time.perf_counter()-begin;self.rows.append(row);return result
    def readout(self,model,batch,frame_ids,records,fit,query,core,torch,**kw):
        torch.cuda.synchronize();t=time.perf_counter()
        if self.enabled:out=full_prefix_readout(self.base,frame_ids,records,fit['final_state'])
        else:out=self.original_readout(model,batch,frame_ids,records,fit,query,core,torch,**kw)
        torch.cuda.synchronize();self.current['final_readout_seconds']=time.perf_counter()-t;self.current['readout']='full_original_both_decoder_passes_from_exact_frozen_views' if self.enabled else 'original_complete_model_forward';return out
    def release(self):
        import torch
        self.base=None;self.current=None;self.number+=1
        if self.number%8==0:gc.collect()
        if torch.cuda.is_initialized():
            free,_=torch.cuda.mem_get_info()
            if free<6*2**30 or torch.cuda.memory_reserved()>24*2**30:gc.collect();torch.cuda.empty_cache()
    def receipt(self):
        write(BASE/'student_profiles'/(self.arm+'.json'),dict(status='complete',arm=self.arm,rows=self.rows,actual_new_source_captures=sum(not r['cache_hit'] for r in self.rows),source_prefix_restores=sum(r['cache_hit'] for r in self.rows),full_prefix_readouts=sum(r.get('readout')=='full_original_both_decoder_passes_from_exact_frozen_views' for r in self.rows),scientific_update_math_unchanged=True,GT_read=False))
