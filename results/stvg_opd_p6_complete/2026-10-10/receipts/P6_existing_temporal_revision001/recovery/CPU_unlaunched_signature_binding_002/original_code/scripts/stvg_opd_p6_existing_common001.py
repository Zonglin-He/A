"""Separate P6 metadata, complete matched original inputs and audit algebra."""
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,PYTHON,read,write,status,sha
from scripts.decota_matrix_common_v1 import save,load
NS=BASE/'P6_existing_temporal_revision001'
ARMS=['Native','ActionnessProjection','TemporalHead','SpatialOPD']
OWN=['scripts/stvg_opd_p6_existing_common001.py','scripts/run_stvg_opd_p6_existing_temporal001.py',
     'scripts/continue_stvg_opd_p6_existing_temporal001.py','scripts/score_stvg_opd_p6_existing_temporal001.py',
     'scripts/stvg_opd_p6_postseal_root001.py',
     'protocols/stvg_opd_p6_existing_temporal001.md']


def config(ds):
    from methods.decota_final_simplified_v1.config import MethodConfig
    return MethodConfig.for_direction('vid_to_hc1' if ds=='hc2' else 'hc2_to_vid')


def definition(ds):return read(BASE/'LATER_DESIGN_LOCK.json')['stages']['P6_'+ds]


def prepare():
    activate()
    assert read(BASE/'P5_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert read(BASE/'P5_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json')['status']=='pass'
    assert read(BASE/'P5_COMPLETE_CLOSING_GITHUB_RECEIPT.json')['status']=='pass'
    if (NS/'REVISION_RUNTIME.json').exists():return verify()
    pins={f:sha(ROOT/f) for f in OWN}
    for p in (ROOT/'methods/decota_final_simplified_v1').glob('*.py'):pins[str(p.relative_to(ROOT))]=sha(p)
    for f in ['scripts/run_spatial_ssl_gpu_v1.py','scripts/run_decota_paper_main_v1.py',
        'scripts/stvg_opd_legacy_input_bridge003.py','scripts/run_decota_corrective_identifiability_v1.py',
        'scripts/score_stvg_opd_paper_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/review_stvg_opd_revised_p1_v2.py',
        'scripts/review_stvg_opd_revised_p5_readback_revision004.py',
        'external/TA-STVG/models/net_utils.py','methods/CURRENT_METHOD.json']:
        pins[f]=sha(ROOT/f)
    d={ds:definition(ds) for ds in ['hc2','vidstg']}
    for ds,s in d.items():
        assert s['conditions']==['clean'] and len(s['parents'])==32 and set(s['orders'])=={'order1'}
        assert set(s['orders']['order1'])==set(s['parents']) and not s['new_temporal_branch']
        assert s['source_selection_without_current_P0_GT']
    write(NS/'REVISION_RUNTIME.json',dict(status='pinned_original_P6_implementation',pins=pins,
        design_sha256=sha(BASE/'LATER_DESIGN_LOCK.json'),CPU_contracts_sha256=sha(NS/'CPU_CONTRACTS.json'),
        CPU_original_input_preflight_sha256=sha(NS/'CPU_ORIGINAL_INPUT_PREFLIGHT.json'),previous_P5_root_sha256=sha(BASE/'P5_ROOT_CLOSING_RECEIPT.json'),
        previous_P5_final_archive_sha256=sha(BASE/'P5_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json'),
        previous_P5_complete_closing_public_sha256=sha(BASE/'P5_COMPLETE_CLOSING_GITHUB_RECEIPT.json'),
        stages=d,temporal_configs={ds:config(ds).to_dict() for ds in d},deployment_arms=ARMS,
        bootstrap=dict(replicates=10000,seed=20261006,unit='paired original parent within each target',multiplicity_adjusted=False),
        offline_GT_head=dict(after_all_deployment_sealed=True,temporal_span='same original official truth span used by existing scorer',
                            legal_pair='first maximum physical tIoU per original offset',same_steps_and_shrink=True,fair_deployment_baseline=False),
        logical_deployment_outputs=256,formal_queries=64,GT_until_global_seal=False,
        new_scientific_algorithm=False,main_method_changed=False,paper_suite_complete=False,time=time.time()))
    return verify()


def verify():
    activate();r=read(NS/'REVISION_RUNTIME.json')
    assert sha(BASE/'LATER_DESIGN_LOCK.json')==r['design_sha256']
    assert sha(NS/'CPU_CONTRACTS.json')==r['CPU_contracts_sha256']
    assert sha(NS/'CPU_ORIGINAL_INPUT_PREFLIGHT.json')==r['CPU_original_input_preflight_sha256']
    assert sha(BASE/'P5_ROOT_CLOSING_RECEIPT.json')==r['previous_P5_root_sha256']
    assert sha(BASE/'P5_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json')==r['previous_P5_final_archive_sha256']
    assert sha(BASE/'P5_COMPLETE_CLOSING_GITHUB_RECEIPT.json')==r['previous_P5_complete_closing_public_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    for ds in ['hc2','vidstg']:assert definition(ds)==r['stages'][ds] and config(ds).to_dict()==r['temporal_configs'][ds]
    return r


def original(ds,query):
    """Read an actual original P0 output and full hash-bound original input."""
    import torch
    from scripts.stvg_opd_paper_common_v1 import digest
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    d=read(BASE/'DESIGN_LOCK.json');stage=d['stages']['P0_'+ds]
    at=stage['orders']['order1'].index(query)
    p=BASE/'stages'/('P0_'+ds)/'clean/order1/on_policy'/f'{at:05}.pt'
    b=read(BASE/'stages'/('P0_'+ds)/'PREDICTION_BARRIER.json');rc=read(p.with_suffix('.json'))
    h=sha(p);assert b['status']=='sealed' and not b['GT_read'] and h==b['files'][str(p.relative_to(BASE))]==rc['sha256']
    assert p.stat().st_size==rc['bytes'] and not rc['GT_read']
    z=load(p);assert z['dataset']==ds and z['query_ordinal']==query and z['arrival']==at
    assert z['arm']=='on_policy' and z['condition']=='clean' and z['order']=='order1' and not z['GT_read']
    assert z['config']==d['datasets'][ds]['config']
    ip=BASE/z['input']['path'];ih=sha(ip);irc=read(ip.with_suffix('.json'))
    assert ih==z['input']['sha256']==irc['sha256'] and not irc['GT_read']
    expected_runtime=read(ROOT/b['original_barrier_path'])['runtime_lock_sha256'] if b.get('exact_stage_reused') else b['runtime_lock_sha256']
    assert irc['runtime_lock_sha256']==expected_runtime
    native=load(ip);n=normalized(native)
    assert n['dataset']==ds and n['parent']==query
    assert n['interval']==z['interval'] and n['corruption_spec'] is None
    assert n['native_boxes'].shape==z['fit']['before'].shape==z['fit']['final'].shape==(len(n['frame_ids']),4)
    assert n['source_model_state_sha256'] in ['fbb1ed8871d6c2aa093879efefc2ee500bb7de5e0fe1c25b809c5393d010f3c7','ee72f0d9a50c573a115bd7cfc2329c860745a1cf3dab83af1be7787c11b3c218']
    prev=None if at==0 else sha(p.parent/f'{at-1:05}.pt')
    assert prev==z['previous_payload_sha256']
    # Interpret exact preserved arrays through their original physical interval;
    # do not write or migrate the original payload/receipt schema.
    z={**z,'frame_ids':n['frame_ids'],'predictions':{label:dict(boxes=boxes,physical_interval=z['interval'],indices=n['indices'])
        for label,boxes in [('Frozen',n['native_boxes']),('Before',z['fit']['before']),('After',z['fit']['final'])]}}
    return z,n,dict(path=str(p.relative_to(BASE)),sha256=h,bytes=p.stat().st_size,
        original_arrival=at,previous_payload_sha256=prev,input_path=str(ip.relative_to(BASE)),input_sha256=ih,
        input_receipt_sha256=sha(ip.with_suffix('.json')),original_input_runtime_sha256=expected_runtime,
        original_barrier_sha256=sha(BASE/'stages'/('P0_'+ds)/'PREDICTION_BARRIER.json'),
        full_expert_pack_sha256=digest(n['expert']))


def array(x):
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    a=np.asarray(x,dtype=np.float64);assert np.isfinite(a).all();return a


def gamma(n):
    u=2**-24;assert 0<n*u<1
    return n*u/(1-n*u)


def lse(x):
    x=array(x);m=x.max();return m+np.log(np.exp(x-m).sum())


def loss_gradient(logits,evidence,margin):
    grads=[];loss=0.;records=[]
    for z,ev in zip(logits,evidence['offsets']):
        z=array(z).reshape(-1,2);ij=np.asarray(ev['ij'],dtype=np.int64);at=int(ev['target'])
        assert np.array_equal(ij,np.asarray(np.triu_indices(len(z),1))) and 0<=at<ij.shape[1]
        sc=z[ij[0],0]+z[ij[1],1];lp=sc-lse(sc);p=np.exp(lp)
        other=lp.copy();other[at]=-np.inf;c=int(np.argmax(other));hinge=max(0.,margin-(lp[at]-lp[c])) if len(lp)>1 else 0.
        loss+=.5*(-lp[at]+hinge);d=.5*p;d[at]-=.5
        if hinge>0:d[at]-=.5;d[c]+=.5
        g=np.zeros_like(z);np.add.at(g[:,0],ij[0],d);np.add.at(g[:,1],ij[1],d)
        grads.append(g);records.append(dict(target=at,competitor=c,hinge_active=hinge>0))
    assert len(grads)==2
    return loss,grads,records


def check_projection(z,actions,records,ev,cfg):
    errors=[]
    for logits,action,record,t in zip(z,actions,records,ev['offsets']):
        v=array(logits).reshape(-1,2);x=np.asarray(array(action),dtype=np.float32).astype(np.float64).reshape(-1)
        f=np.asarray(record['frame_ids']);assert len(f)==len(v)==len(x) and np.all(np.diff(f)>0)
        ij=np.asarray(np.triu_indices(len(v),1));lp=v[ij[0],0]+v[ij[1],1];lp-=lse(lp)
        med=np.quantile(x,.5);mad=np.quantile(np.abs(x-med),.5);u=(x-cfg.center_fraction*med)/(mad+cfg.epsilon)
        edges=np.r_[f[0],(f[:-1]+f[1:])/2,f[-1]+1];w=np.diff(edges)/(edges[-1]-edges[0])
        ln=-np.logaddexp(0.,u);prefix=np.r_[0.,np.cumsum(w*u)];cost=-np.sum(w*ln)-(prefix[ij[1]+1]-prefix[ij[0]])
        score=-cost+cfg.prior_weight*lp;at=int(score.argmax())
        assert at==t['target'] and [int(ij[0,at]),int(ij[1,at])]==t['target_indices']
        assert t['interval']==[int(f[ij[0,at]]),int(f[ij[1,at]]+1)]
        for name,a in [('logp0',lp),('score',score),('cost',cost),('raw_logits',x),('standardized_logits',u),('omega',w),('cell_edges',edges)]:
            actual=array(t[name]);error=float(np.max(np.abs(actual-a)));assert np.all(np.abs(actual-a)<=1e-9*(1+np.abs(a))),(name,error)
            errors.append(error)
    return max(errors)


def audit(fit,record,records,cfg,oracle=False):
    """NumPy algebra from same actual values; no model or optimizer calls."""
    import torch
    ev=fit['evidence'];evaluations=record['evaluations'];steps=record['steps']
    assert len(evaluations)==cfg.temporal_steps+2 and len(steps)==cfg.temporal_steps
    assert fit['failure'] is None and not fit['skipped'] and fit['selected_step']==cfg.temporal_steps
    assert fit['backwards']==cfg.temporal_steps and fit['evaluations']==cfg.temporal_steps+1
    assert fit['eta']==cfg.eta and fit['GT_online'] is False and fit['source_restored']
    assert sum(v.numel() for v in fit['initial_state'].values())==66306
    projection=0. if oracle else check_projection(evaluations[0]['logits'],record['actions'],records,ev,cfg)
    maximum=dict(projection=projection,loss=0.,native_logit_gradient=0.,head_forward=0.,head_gradient=0.,Adam_m=0.,Adam_v=0.,Adam_parameter=0.)
    def bounded(name,actual,expected,scale,n=64):
        actual=array(actual);expected=array(expected);err=float(np.max(np.abs(actual-expected)))
        assert actual.shape==expected.shape and np.all(np.abs(actual-expected)<=2e-5+gamma(n)*np.asarray(scale)),(name,err)
        maximum[name]=max(maximum[name],err)
    for i,e in enumerate(evaluations[:cfg.temporal_steps+1]):
        state=e['state'];old=fit['initial_state'] if i==0 else steps[i-1]['after']
        assert all(torch.equal(state[k],old[k].cpu()) for k in state)
        assert all(torch.equal(state[k],fit['path'][i]['state'][k].cpu()) for k in state)
        loss,gg,details=loss_gradient(e['logits'],ev,cfg.margin)
        err=abs(loss-fit['losses'][i]);assert err<=1e-9*(1+abs(loss));maximum['loss']=max(maximum['loss'],err)
        if i==cfg.temporal_steps:continue
        st=steps[i];assert st['step']==i+1 and len(e['linears'])==4
        assert all(torch.equal(st['before'][k],state[k]) for k in state)
        sums={k:np.zeros_like(array(v)) for k,v in state.items()};scales={k:np.zeros_like(array(v)) for k,v in state.items()}
        for j in range(2):
            native=array(e['native_logit_gradients'][j]).reshape(-1,2)
            bounded('native_logit_gradient',native,gg[j],np.abs(gg[j]),2*len(gg[j])+32)
            a=e['linears'][j*2];b=e['linears'][j*2+1];x=array(a['input']).reshape(-1,256);h=array(b['input']).reshape(-1,256)
            for linear,layer in [(a,0),(b,1)]:
                xx=array(linear['input']).reshape(-1,256);ww=array(state[f'head.layers.{layer}.weight']);bb=array(state[f'head.layers.{layer}.bias'])
                bounded('head_forward',array(linear['output']).reshape(-1,len(bb)),xx@ww.T+bb,np.abs(xx)@np.abs(ww.T)+np.abs(bb),2*256+8)
            assert np.array_equal(h,np.maximum(array(a['output']).reshape(-1,256),0))
            dy=native;dh=dy@array(state['head.layers.1.weight']);dh*=array(a['output']).reshape(-1,256)>0
            vv={'head.layers.1.weight':dy.T@h,'head.layers.1.bias':dy.sum(0),
                'head.layers.0.weight':dh.T@x,'head.layers.0.bias':dh.sum(0)}
            ss={'head.layers.1.weight':np.abs(dy).T@np.abs(h),'head.layers.1.bias':np.abs(dy).sum(0),
                'head.layers.0.weight':np.abs(dh).T@np.abs(x),'head.layers.0.bias':np.abs(dh).sum(0)}
            for k in vv:sums[k]+=vv[k];scales[k]+=ss[k]
        for k in state:
            g=array(st['gradient'][k]);bounded('head_gradient',g,sums[k],scales[k],2*(256+sum(len(v) for v in gg))+32)
            prev=st['before_opt'][k];after=st['after_opt'][k]
            if i==0:assert prev is None
            else:
                assert prev['step']==i and all(torch.equal(prev[v],steps[i-1]['after_opt'][k][v]) for v in ['m','v'])
            m0=np.zeros_like(g) if prev is None else array(prev['m']);v0=np.zeros_like(g) if prev is None else array(prev['v'])
            m=.9*m0+.1*g;v=.999*v0+.001*g*g;assert after['step']==i+1
            bounded('Adam_m',after['m'],m,np.abs(m0)+np.abs(g),16)
            bounded('Adam_v',after['v'],v,np.abs(v0)+g*g,16)
            # Use the actual same-step rounded GPU moments for independent update algebra.
            mm=array(after['m']);vv=array(after['v']);delta=-cfg.temporal_lr*(mm/(1-.9**(i+1)))/(np.sqrt(vv/(1-.999**(i+1)))+1e-4)
            expected=array(st['before'][k])+delta
            bounded('Adam_parameter',st['after'][k],expected,np.abs(array(st['before'][k]))+np.abs(delta),64)
    assert all(torch.equal(fit['state'][k].cpu(),steps[-1]['after'][k]) and torch.equal(fit['last_state'][k].cpu(),steps[-1]['after'][k]) for k in fit['state'])
    for k,v in fit['initial_state'].items():
        expected=v.cpu()+cfg.eta*(fit['state'][k].cpu()-v.cpu())
        assert torch.equal(expected,fit['shrunk_state'][k].cpu()) and torch.equal(expected,evaluations[-1]['state'][k])
    return dict(status='pass',scope='same actual projection/loss/native-head-output VJP/two-layer head gradient/AdamW/66306 state/reset/last/shrink',
        evaluations=len(evaluations),backwards=len(steps),state_coordinates=66306*(cfg.temporal_steps+1),
        maximum_errors=maximum,CPU64_formula_and_declared_float32_operation_bounds=True,
        independent_full_decoder_Jacobian=False,CUDA_transcendental_proof=False,GT_read=bool(oracle))
