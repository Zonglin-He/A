"""Prepare source-only storages, then strictly chronological external-state TTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,hashlib,zipfile,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,save,load
from vg_tta.stvg_external_opd20_v1 import ARMS,CONFIG
D=ROOT/'artifacts/stvg_external_opd20_online64_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/external_opd20_online64'
PRE=ROOT/'artifacts/stvg_correction_distribution64_v1'


def status(stage,done,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,done=done,total=64,pid=os.getpid(),timestamp=time.time(),**kw),True)


def npz_save(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();tmp=path.with_suffix('.tmp')
    with tmp.open('wb')as f:np.savez(f,**values)
    tmp.replace(path);r=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size);put(path.with_suffix('.receipt.json'),r);return r


def source_storages(path,ids):
    """Read only immutable serializer's first Source/logits storages; NO pickle."""
    n=len(ids);reads=[];payloads=[]
    with zipfile.ZipFile(path)as z:
        prefix=z.namelist()[0].split('/')[0];name=prefix+'/byteorder';assert z.read(name)==b'little';reads.append(name)
        for k,shape in [(0,(n,4)),(1,(1,len(ids[::2]),2)),(2,(1,len(ids[1::2]),2))]:
            member=f'{prefix}/data/{k}';raw=z.read(member);reads.append(member)
            assert len(raw)==4*int(np.prod(shape)),(path,member,shape,len(raw))
            a=np.frombuffer(raw,dtype='<f4').reshape(shape).copy();assert np.isfinite(a).all();payloads.append(a)
    assert not any('data.pkl'in p for p in reads)
    return payloads,dict(members_read=reads,payload_sha256=[hashlib.sha256(a.tobytes()).hexdigest()for a in payloads],no_pickle_or_expert_payload_read=True)


def prepare():
    import torch
    from vg_tta.stvg_external_opd20_v1 import contract
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.observations import uniform_positions
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    assert not(D/'PLAN_LOCK.json').exists();old=read(PRE/'PLAN_LOCK.json');records=[]
    for ds in ['hc2','vidstg']:
        rr=[r for r in old['records']if r['dataset']==ds];assert len(rr)==32
        seed=int(hashlib.sha256(('external-opd20-online64-v1|20261018|'+ds).encode()).hexdigest()[:16],16)
        order=np.random.Generator(np.random.PCG64(seed)).permutation(32).tolist()
        for i in range(16):
            for role,j in [('expert',order[i]),('no_expert',order[16+i])]:
                r=rr[j];records.append(dict(alias=r['alias'],dataset=ds,parent=r['parent'],query_ordinal=r['query_ordinal'],role=role,domain_arrival=2*i+(role=='no_expert'),
                    source_path=str((D/'source_only'/(r['alias']+'.npz')).relative_to(ROOT)),
                    expert_path=str((D/'expert_only'/(r['alias']+'.npz')).relative_to(ROOT))if role=='expert'else None))
    put(D/'PLAN_LOCK.json',dict(status='locked_before_input_export_updates_and_new_GT',records=records,count=64,arms=ARMS,
        exact_prior64_plan_sha256=sha(PRE/'PLAN_LOCK.json'),split_seed=20261018,split_rule='Domain namespace SHA256-seeded PCG64 permutation; first16 expert, remaining16 no-expert; interleave E,N',
        configurations=CONFIG,external_variables=20,state='Independent zero W per branch/domain, W only carry, reset Adam per expert',
        primary='no_expert arrivals16 per domain, saved-history conditional paired uncertainty',
        expert_final_output='identical fixed E all five arms',source_scale='existing opposite-source E geometric scale',
        new_backbone_calls=0,new_DINO_calls=0,backbone_updates=0,new_Temporal_updates=0,source_training=0,
        historical_development_exposure=True,independent_confirmation=False,max_expert_fit_slots=128,max_external_optimizer_steps=640,
        CPU_measure_cap_seconds=1800,created_epoch=time.time(),deadline_epoch=time.time()+7200))
    by={r['alias']:r for r in old['records']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']}
    calibration=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');srcfiles=[];expertfiles=[];layout=[]
    for r in records:
        oldr=by[r['alias']];ids=plans[r['dataset']][r['query_ordinal']]['frame_ids'];payloads,trace=source_storages(ROOT/oldr['prior_Source'],ids);base,l0,l1=payloads
        rec=[dict(frame_ids=ids[o::2])for o in [0,1]];g=probabilities(rec,[l0,l1]);outids,q=mass_table(g);assert outids==ids;interval=final_mbr(ids,q)['interval']
        native=prediction([torch.from_numpy(l0),torch.from_numpy(l1)],torch.from_numpy(base),rec,ids)
        positions=uniform_positions(ids,native['indices'],4)
        scale=np.array(calibration['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2']['scale'])
        srcfiles.append(npz_save(ROOT/r['source_path'],boxes=base.astype(np.float64),frame_ids=np.array(ids,dtype=np.int64),interval=np.array(interval,dtype=np.int64),planned_positions=np.array(positions,dtype=np.int64),scale=scale))
        if r['role']=='expert':
            assert sha(ROOT/oldr['prior_prediction'])==oldr['prediction_sha256'];z=load(ROOT/oldr['prior_prediction'])
            assert np.array_equal(base.astype(np.float64),z['outputs']['Source_no_correction'])and z['frame_ids']==ids
            assert z['interval']==interval and z['planned_positions']==positions
            expertfiles.append(npz_save(ROOT/r['expert_path'],teacher=z['outputs']['E_DINO_support'],active=np.array(z['estimate']['support_frames']>0),planned_positions=np.array(positions,dtype=np.int64)))
            trace['independent_expert_Source_storage_time_positions_exact']=True
        layout.append(dict(alias=r['alias'],role=r['role'],**trace))
    own=['vg_tta/stvg_external_opd20_v1.py','scripts/run_stvg_external_opd20_online64_v1.py','protocols/stvg_external_opd20_online64_v1.md',
        'scripts/stvg_temporal_marginal_decode_util256_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py','methods/decota_final_simplified_v1/objectives.py','methods/decota_final_simplified_v1/observations.py',
        'scripts/run_stvg_dino_supported_source256_v1.py']
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},plan_sha256=sha(D/'PLAN_LOCK.json'),registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}))
    put(D/'INPUT_BARRIER.json',dict(status='complete',count=64,source_only=srcfiles,expert_only=expertfiles,expert_inputs=32,no_expert_inputs=32,layout_checks=layout,
        no_expert_E_or_DINO_exported_or_deserialized=False if any(r['role']=='no_expert'and r['expert_path']is not None for r in records)else True,GT_read=False))
    put(D/'CPU_CONTRACT_COMPLETION.json',contract());status('prepared',0,GT_read=False)
    assert not torch.cuda.is_initialized();print(json.dumps(dict(status='prepared',sources=64,expert=32,no_expert=32,variables=20)))


def verify():
    from main_method.runtime import resource_guard
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert time.time()<p['deadline_epoch']and sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    for f,h in {**l['pins'],**l['registries']}.items():assert sha(ROOT/f)==h,f
    resource_guard();return p


def measure():
    import torch
    from vg_tta.stvg_external_opd20_v1 import update,phi,transform,snapshot
    torch.set_num_threads(4);p=verify();inputs=read(D/'INPUT_BARRIER.json');assert inputs['count']==64
    source_sha={r['path']:r['sha256']for r in inputs['source_only']};expert_sha={r['path']:r['sha256']for r in inputs['expert_only']}
    current=dict(alias=None,role=None,path=None);reads=[]
    def access_guard(event,args):
        if event!='open'or not args or not isinstance(args[0],(str,bytes,Path)):return
        v=os.fsdecode(args[0]);a=str(Path(v).absolute())
        if '/expert_only/'in a and a.endswith('.npz'):
            assert current['role']=='expert'and a==current['path'],'Forbidden expert input outside current expert arrival'
            reads.append(dict(alias=current['alias'],role=current['role'],path=str(Path(a).relative_to(ROOT))))
        if any(x in a for x in ['/annos/','/annotations/','/checkpoints/','/model_zoo/']):raise PermissionError('Online worker forbids GT/model asset reads')
        if a.startswith(str(ROOT/'results')+'/'):raise PermissionError('Online worker forbids prior results')
        for old in ['stvg_dino_supported_source256_v1','stvg_correction_distribution64_v1','stvg_opd_paper_hc2_revision_v2']:
            if a.startswith(str(ROOT/'artifacts'/old)+'/'):raise PermissionError('Online worker forbids original mixed Source/E/DINO artifacts')
    sys.addaudithook(access_guard)
    Ws={};domain=None;predecessor=None;files=[];started=time.perf_counter();allstats=[]
    for i,r in enumerate(p['records']):
        verify();assert time.perf_counter()-started<p['CPU_measure_cap_seconds']
        if domain!=r['dataset']:
            domain=r['dataset'];Ws={a:torch.zeros((4,5),dtype=torch.float64,requires_grad=a!='A_no_learning')for a in ARMS};predecessor=None
        current.update(alias=r['alias'],role=r['role'],path=str(ROOT/r['expert_path'])if r['expert_path']else None)
        assert sha(ROOT/r['source_path'])==source_sha[r['source_path']]
        with np.load(ROOT/r['source_path'],allow_pickle=False)as z:
            assert set(z.files)=={'boxes','frame_ids','interval','planned_positions','scale'}
            base=torch.tensor(z['boxes'],dtype=torch.float64);ids=z['frame_ids'].tolist();interval=z['interval'].tolist();planned=z['planned_positions'].tolist();scale=torch.tensor(z['scale'])
        Wbefore={a:w.detach().clone()for a,w in Ws.items()};fits={};outputs={};active=None;target=None
        tick=time.perf_counter()
        if r['role']=='expert':
            assert sha(ROOT/r['expert_path'])==expert_sha[r['expert_path']]
            with np.load(ROOT/r['expert_path'],allow_pickle=False)as z:
                assert set(z.files)=={'teacher','active','planned_positions'};target=torch.tensor(z['teacher'],dtype=torch.float64);active=bool(z['active']);assert z['planned_positions'].tolist()==planned
            for a in ARMS[1:]:fits[a]=update(Ws[a],base,scale,target,planned,a,domain+'|'+r['alias'],active)
            outputs={a:target.numpy().copy()for a in ARMS}
        else:
            assert r['expert_path']is None
            with torch.no_grad():outputs={a:transform(base,w@phi(base),scale).numpy().copy()for a,w in Ws.items()}
            assert all(torch.equal(w,Wbefore[a])for a,w in Ws.items())
        after={a:w.detach().clone()for a,w in Ws.items()};assert torch.count_nonzero(Ws['A_no_learning'])==0
        out=dict(alias=r['alias'],dataset=domain,role=r['role'],domain_arrival=r['domain_arrival'],frame_ids=ids,interval=interval,planned_positions=planned,Source_boxes=base.numpy(),scale=scale.numpy(),phi=phi(base).numpy(),
            W_before=Wbefore,W_after=after,outputs=outputs,fits=fits,valid_expert=active,teacher=None if target is None else target.numpy(),
            predecessor_sha256=predecessor,source_input_sha256=source_sha[r['source_path']],expert_input_sha256=expert_sha[r['expert_path']]if r['expert_path']else None,
            GT_read=False,compute=dict(CPU_arrival_seconds=time.perf_counter()-tick,new_model_calls=0,new_DINO_calls=0,external_optimizer_steps=sum(f['actual_steps']for f in fits.values())))
        rec=save(D/'predictions'/(r['alias']+'.pt'),out);files.append(rec);predecessor=rec['sha256'];allstats.append(dict(alias=r['alias'],role=r['role'],steps=out['compute']['external_optimizer_steps']))
        status('online_external_state',i+1,GT_read=False);print(domain,r['domain_arrival'],r['role'],r['alias'],flush=True)
    current.update(alias=None,role=None,path=None)
    assert not torch.cuda.is_initialized()
    put(D/'ACCESS_READBACK.json',dict(status='pass',count=64,expert_open_events=reads,nonexpert_teacher_opens=0,original_mixed_payload_opens=0,backbone_or_GT_asset_opens=0,future_expert_input_hashing_not_performed=True))
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=files,all64_five_arm_outputs_before_GT=True,GT_read=False,CPU_wall_seconds=time.perf_counter()-started,new_model_calls=0,new_DINO_calls=0,backbone_updates=0,external_optimizer_steps=sum(r['steps']for r in allstats)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure']);a=p.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error=str(e),partial_preserved=True));raise
