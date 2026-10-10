"""Finite original P6 temporal-head qualification/deployment, never target GT."""
import argparse
import collections
import copy
import gc
import os
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import *


class Recorder:
    """Observation-only hooks on the original optimizer and head; no new update."""
    def __init__(self,replay):
        import torch
        from methods.decota_final_simplified_v1.tensors import detached
        self.replay=replay;self.original_values=replay.values;self.original_cls=torch.optim.AdamW
        self.data=dict(evaluations=[],steps=[],actions=detached(replay.zero['actions'],'cpu'))
        self.handles=[];self.current=None;self.pending=None
        def linear_hook(module,args,out):
            if self.current is not None:
                self.current['linears'].append(dict(input=detached(args[0][-1],'cpu'),output=detached(out[-1],'cpu')))
        for linear in replay.head.layers:self.handles.append(linear.register_forward_hook(linear_hook))
        def values():
            e=dict(state=detached(replay.state(),'cpu'),linears=[],native_logit_gradients=[None,None])
            self.current=e
            try:v=self.original_values()
            finally:self.current=None
            e['logits']=detached(v['logits'],'cpu');self.data['evaluations'].append(e)
            for j,z in enumerate(v['logits']):
                if z.requires_grad:
                    def keep(g,j=j,e=e):e['native_logit_gradients'][j]=detached(g,'cpu')
                    z.register_hook(keep)
            return v
        def state(opt,p):
            s=opt.state.get(p,{})
            if not s:return None
            return dict(step=int(s['step']),m=detached(s['exp_avg'],'cpu'),v=detached(s['exp_avg_sq'],'cpu'))
        def before(opt,args,kwargs):
            self.pending=dict(before=detached(replay.state(),'cpu'),
                gradient={n:detached(p.grad,'cpu') for n,p in replay.named},
                before_opt={n:state(opt,p) for n,p in replay.named})
        def after(opt,args,kwargs):
            s=self.pending;s.update(after=detached(replay.state(),'cpu'),after_opt={n:state(opt,p) for n,p in replay.named},step=len(self.data['steps'])+1)
            self.data['steps'].append(s);self.pending=None
        def constructor(*args,**kwargs):
            opt=self.original_cls(*args,**kwargs)
            self.handles.extend([opt.register_step_pre_hook(before),opt.register_step_post_hook(after)])
            return opt
        self.values=values;self.constructor=constructor

    def __enter__(self):
        import torch
        self.replay.values=self.values;torch.optim.AdamW=self.constructor
        return self

    def __exit__(self,*args):
        import torch
        self.replay.values=self.original_values;torch.optim.AdamW=self.original_cls
        for h in self.handles:h.remove()


def recorded_fit(replay,records,cfg,oracle=None):
    from methods.decota_final_simplified_v1.optim import fit_temporal,_fit
    from methods.decota_final_simplified_v1.objectives import temporal_loss
    from methods.decota_final_simplified_v1.tensors import detached
    import torch
    with Recorder(replay) as rec:
        if oracle is None:f=fit_temporal(replay,records,cfg,trace=True)
        else:
            f=_fit(replay,replay.zero,lambda v:temporal_loss(v['logits'],oracle,cfg.margin),lr=cfg.temporal_lr,steps=cfg.temporal_steps,temporal=True,trace=True)
            f['evidence']=oracle
            state={k:v+cfg.eta*(f['state'][k]-v) for k,v in replay.initial.items()}
            try:
                replay.restore(state)
                with torch.no_grad():final=detached(replay.values())
            finally:replay.restore(replay.initial)
            f.update(shrunk_state=state,shrunk=final,eta=cfg.eta,shrunk_parameter_changed=any(not torch.equal(state[k],v) for k,v in replay.initial.items()))
    return f,rec.data


def equal(a,b):
    import torch
    if torch.is_tensor(a):assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


def fit_once(model,row,frames,original_input,cfg,recorded):
    import torch
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from methods.decota_final_simplified_v1.replay import TemporalReplay
    from methods.decota_final_simplified_v1.optim import fit_temporal
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    batch,records,base=frozen_forward(model,frames,row);ids=row['frame_ids']
    native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
    assert torch.equal(native['boxes'],original_input['native_boxes'])
    assert native['physical_interval']==original_input['interval'] and ids==original_input['frame_ids']
    assert state_hash(model.state_dict())==original_input['source_model_state_sha256']==model._fixed_full_state_hash
    replay=TemporalReplay(model.temp_embed,base.temporal_inputs,base.zero)
    if recorded:fit,trajectory=recorded_fit(replay,records,cfg)
    else:fit=fit_temporal(replay,records,cfg,trace=True);trajectory=None
    assert fit['failure'] is None and fit['selected_step']==5 and fit['backwards']==5
    before=time.perf_counter()
    with query_subject(model,batch,row['parses']['subject']):
        final=full_prediction(model,batch,ids,records,{**base.initial,**fit['shrunk_state']},fit['shrunk'])
    reinsert_seconds=time.perf_counter()-before
    assert torch.equal(final['boxes'],native['boxes']) and state_hash(model.state_dict())==model._fixed_full_state_hash
    evidence=fit['evidence'];intervals=[e['interval'] for e in evidence['offsets']]
    project_interval=[min(v[0] for v in intervals),max(v[1] for v in intervals)]
    project_pred=dict(boxes=native['boxes'].clone(),physical_interval=project_interval,raw_physical_intervals=intervals,
        raw_offset_indices=[e['target_indices'] for e in evidence['offsets']],indices=[ids.index(project_interval[0]),ids.index(project_interval[1]-1)])
    capture=dict(records=records,frame_ids=ids,zero=detached(base.zero,'cpu'),temporal_inputs=detached(base.temporal_inputs,'cpu'),
        source_head_state=detached(replay.initial,'cpu'),source_model_state_sha256=model._fixed_full_state_hash)
    result=dict(fit=detached(fit,'cpu'),predictions=dict(Native=native,ActionnessProjection=project_pred,TemporalHead=final),capture=capture)
    if recorded:
        result['trajectory']=trajectory
        try:result['math_audit']=audit(result['fit'],trajectory,records,cfg)
        except BaseException:
            witness=NS/'unsealed_failed_fit_witnesses'/(str(time.time_ns())+'.pt')
            save(witness,result)
            write(witness.with_suffix('.json'),dict(status='original_complete_fit_preserved_before_exit',path=str(witness.relative_to(BASE)),
                sha256=sha(witness),bytes=witness.stat().st_size,accepted_formal_predictions=0,GT_read=False,
                traceback=traceback.format_exc(),time=time.time()))
            raise
    result['cost']=dict(full_reinsertion_seconds=reinsert_seconds)
    del replay,base,batch
    return result


def qualification(ds):
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_opd_paper_common_v1 import budget
    verify();folder=NS/'qualification'/ds;cfg=config(ds);seq=definition(ds)['orders']['order1'][:2]
    assert not (folder/'GPU_QUALIFICATION.json').exists()
    with gpu():
        model=model_for(definition(ds)['source']);cache=collections.OrderedDict();records=[]
        for at,q in enumerate(seq):
            budget();orig,inp,binding=original(ds,q);row=read_row(ds,q)
            frames,ids,reuse=frames_for(ds,row,cache);shifted,pixel,spec=observation(row,'clean',frames)
            assert pixel==inp['pixel_sha256'] and spec is None and ids==inp['frame_ids']
            raw=fit_once(model,row,shifted,inp,cfg,False)
            p0=folder/f'{at:05}_original.pt';save(p0,raw)
            new=fit_once(model,row,shifted,inp,cfg,True)
            for key in ['fit','predictions','capture']:equal(raw[key],new[key])
            p1=folder/f'{at:05}_recorded.pt';save(p1,new)
            assert state_hash(model.state_dict())==model._fixed_full_state_hash
            records.append(dict(arrival=at,query_ordinal=q,original_fit=dict(path=str(p0.relative_to(BASE)),sha256=sha(p0),bytes=p0.stat().st_size),
                recorded_fit=dict(path=str(p1.relative_to(BASE)),sha256=sha(p1),bytes=p1.stat().st_size),complete_scientific_fit_bitwise=True,
                actual_source_process_hash_unchanged=True,original_binding=binding,math_audit=new['math_audit']))
            status(folder/'STATUS.json',dict(status='running_qualification',completed=len(records),expected=2,GT_read=False,process=os.getpid(),time=time.time()))
            del orig,inp,shifted,raw,new;gc.collect()
        write(folder/'GPU_QUALIFICATION.json',dict(status='pass',scope='two original versus recorded complete existing temporal head fits and full reinsertion',
            dataset=ds,arrivals=2,actual_complete_GPU_fits=4,records=records,formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,
            runtime_sha256=sha(RUNTIME),source_process_hash=model._fixed_full_state_hash,time=time.time()))


def run(ds):
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_opd_paper_common_v1 import budget
    verify();assert read(BASE/'P6_QUALIFICATION.json')['status']=='pass'
    cfg=config(ds);stage=definition(ds);folder=NS/'formal'/ds;sequence=stage['orders']['order1']
    qualified=read(NS/'qualification'/ds/'GPU_QUALIFICATION.json');all_bindings=[]
    with gpu():
        model=model_for(stage['source']);cache=collections.OrderedDict()
        for at,q in enumerate(sequence):
            budget();path=folder/f'{at:05}.pt'
            if path.with_suffix('.json').exists():
                old=read(path.with_suffix('.json'));assert old['sha256']==sha(path) and not old['GT_read'];all_bindings.append(old);continue
            assert not path.exists(),'Preserve unreceipted formal payload; root engineering readback required'
            orig,inp,binding=original(ds,q);row=read_row(ds,q);started=time.perf_counter()
            frames,ids,reuse=frames_for(ds,row,cache);shifted,pixel,spec=observation(row,'clean',frames)
            assert pixel==inp['pixel_sha256'] and spec is None and ids==inp['frame_ids']
            capture_start=time.perf_counter();result=fit_once(model,row,shifted,inp,cfg,True)
            elapsed=time.perf_counter()-capture_start
            result['predictions']['SpatialOPD']=orig['predictions']['After']
            result['original_spatial_before']=orig['predictions']['Before']
            result['original_spatial_fit_binding']=binding
            # A diagnostic subset retains its original full-stream predecessor.
            assert result['predictions']['SpatialOPD']['physical_interval']==result['predictions']['Native']['physical_interval']
            if at<2:
                qual=qualified['records'][at];assert qual['query_ordinal']==q
                qp=BASE/qual['recorded_fit']['path'];assert sha(qp)==qual['recorded_fit']['sha256'];control=load(qp)
                for k in ['fit','capture']:equal(result[k],control[k])
                for k in ['Native','ActionnessProjection','TemporalHead']:equal(result['predictions'][k],control['predictions'][k])
                formal_pair=dict(status='pass',qualified_sha256=sha(qp),complete_fit_and_input_bitwise=True)
            else:formal_pair=None
            assert state_hash(model.state_dict())==model._fixed_full_state_hash
            result.update(dataset=ds,source=stage['source'],query_ordinal=q,arrival=at,source_id=row['source'],frame_ids=ids,
                pixel_sha256=pixel,corruption_spec=spec,config=cfg.to_dict(),GT_read=False,new_DINO_calls=0,
                runtime_sha256=sha(RUNTIME),first_formal_bitwise=formal_pair)
            result['cost'].update(complete_capture_fit_audit_reinsertion_seconds=elapsed,complete_query_wall_seconds=time.perf_counter()-started,
                CUDA_peak_allocated=torch.cuda.max_memory_allocated(),new_DINO_calls=0)
            save(path,result);rc=dict(status='accepted_complete_formal_query',dataset=ds,arrival=at,query_ordinal=q,
                path=str(path.relative_to(BASE)),sha256=sha(path),bytes=path.stat().st_size,GT_read=False,outputs=4,
                first_formal_qualified_bitwise=formal_pair,original_spatial_binding=binding,runtime_sha256=sha(RUNTIME),time=time.time())
            write(path.with_suffix('.json'),rc);all_bindings.append(rc)
            status(folder/'STATUS.json',dict(status='running',done=len(all_bindings),expected=32,logical_outputs=4*len(all_bindings),GT_read=False,process=os.getpid(),time=time.time()))
            status(BASE/'stages'/('P6_'+ds)/'STATUS.json',dict(status='running',done=len(all_bindings),expected=32,logical_outputs=4*len(all_bindings),GT_read=False,process=os.getpid(),time=time.time()))
            del orig,inp,result,shifted
            if at<2:del control
            gc.collect()
        assert len(all_bindings)==32
        write(folder/'PREDICTION_BARRIER.json',dict(status='sealed',scope='all four deployment arms on fixed 32 matched queries',dataset=ds,
            queries=32,logical_outputs=128,arms=ARMS,records=all_bindings,GT_read=False,new_DINO_calls=0,
            runtime_sha256=sha(RUNTIME),source_process_hash=model._fixed_full_state_hash,time=time.time()))


def contracts():
    """Actual synthetic CPU optimizers and changed-value rejection, no GPU/GT."""
    import torch
    from methods.decota_final_simplified_v1.tensors import ParameterState,detached
    from scripts.run_decota_corrective_identifiability_v1 import make_head
    from methods.decota_final_simplified_v1.optim import fit_temporal
    class Replica(ParameterState):
        def __init__(self):
            torch.manual_seed(611032)
            from importlib.util import spec_from_file_location,module_from_spec
            sp=spec_from_file_location('P6_contract_MLP',ROOT/'external/TA-STVG/models/net_utils.py');m=module_from_spec(sp);sp.loader.exec_module(m)
            self.head=m.MLP(256,256,2,2,dropout=.3).float().eval()
            self.named=[('head.'+n,p) for n,p in self.head.named_parameters()];self.initial=self.state()
            self.inputs=[torch.randn(6,1,4,256),torch.randn(6,1,4,256)]
            self.zero=dict(boxes=torch.zeros(8,4),logits=[self.head(h)[-1].detach() for h in self.inputs],actions=[torch.tensor([-.2,.8,.7,-.3]),torch.tensor([-.1,.4,.6,-.7])])
        def values(self):return {**self.zero,'logits':[self.head(h)[-1] for h in self.inputs]}
    torch.set_num_threads(4);rows=[];rejected=[];rs=[dict(frame_ids=[0,2,4,6]),dict(frame_ids=[1,3,5,7])]
    for ds in ['hc2','vidstg']:
        rep=Replica();cfg=config(ds);old=fit_temporal(rep,rs,cfg,trace=True);new,rec=recorded_fit(rep,rs,cfg)
        equal(old,new);math=audit(new,rec,rs,cfg)
        rows.append(dict(dataset=ds,status='pass',complete_synthetic_CPU_fit_bitwise=True,math=math))
        for change in ['loss','target','native_gradient','parameter','moment','shrink','path','actions']:
            f=copy.deepcopy(new);r=copy.deepcopy(rec)
            if change=='loss':f['losses'][0]+=.1
            elif change=='target':f['evidence']['offsets'][0]['target']=(f['evidence']['offsets'][0]['target']+1)%6
            elif change=='native_gradient':r['evaluations'][0]['native_logit_gradients'][0].add_(.1)
            elif change=='parameter':r['steps'][0]['after']['head.layers.1.weight'].add_(.1)
            elif change=='moment':r['steps'][0]['after_opt']['head.layers.1.weight']['m'].add_(.1)
            elif change=='shrink':f['shrunk_state']['head.layers.1.weight'].add_(.1)
            elif change=='path':f['path'][1]['state']['head.layers.1.weight'].add_(.1)
            else:r['actions'][0].mul_(-1)
            try:audit(f,r,rs,cfg)
            except (AssertionError,ValueError):rejected.append(dict(dataset=ds,change=change,rejected=True))
            else:raise AssertionError('changed actual contract accepted '+change)
    write(NS/'CPU_CONTRACTS.json',dict(status='pass',scope='actual synthetic CPU original-fit parity and independent complete-head algebra/error rejection',valid=2,invalid_rejected=len(rejected),records=rows,rejections=rejected,CPU_only=True,new_GPU_calls=0,GT_read=False,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['contracts','qualification','formal'],required=True);p.add_argument('--dataset',choices=['hc2','vidstg']);a=p.parse_args()
    try:
        if a.mode=='contracts':contracts()
        elif a.mode=='qualification':qualification(a.dataset)
        else:run(a.dataset)
    except BaseException:
        status(NS/'FAILURE.json',dict(status='failed',scope=a.mode,dataset=a.dataset,process=os.getpid(),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
