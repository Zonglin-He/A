"""Integrated Final online execution, matched frozen/fast/slow controls, no GT."""
import sys,time,gc,traceback,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT as S0,OLD,CONDS,observation
OUT=ROOT/'artifacts/tastvg_matched_ablation_a2_v1';J0=ROOT/'artifacts/tastvg_joint_j0_v1';TEMP=ROOT/'artifacts/tastvg_temporal_fourarm_v1';NATIVE=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1';METHOD=ROOT/'methods/tastvg_dual_evidence_j0_v1'

J01=ROOT/'artifacts/tastvg_schedule_j01_v1'

def prepare():
    assert not (OUT/'LOCK.json').exists(), 'Do not overwrite an existing lock'
    p=read(J01/'LOCK.json');freeze=read(METHOD/'FREEZE_J01.json')
    assert sha(METHOD/'method.py')==freeze['method_code_sha256'] and sha(METHOD/'config.json')==freeze['config_sha256']
    from vg_tta.tastvg_matched_ablation_a2_v1 import ARMS,permutation
    pins={**p['pins'],**{f:sha(ROOT/f) for f in ['protocols/tastvg_matched_ablation_a2_v1.md','scripts/run_tastvg_matched_ablation_a2_v1.py','vg_tta/tastvg_matched_ablation_a2_v1.py','tests/test_tastvg_matched_ablation_a2_v1.py']}}
    assignments={o:{str(r['ordinal']):permutation(r['source'],o) for r in p['rows']} for o in p['orders']}
    write(OUT/'LOCK.json',dict(rows=p['rows'],orders=p['orders'],conditions=CONDS,arms=ARMS,expert_indices=p['expert_indices'],pins=pins,inputs=p['inputs'],freeze_sha256=sha(METHOD/'FREEZE_J01.json'),J01_barrier_sha256=sha(J01/'PREDICTION_BARRIER.json'),permutations=assignments,cap_seconds=2400,time=time.time()))
    write(OUT/'ASSIGNMENTS.json',dict(orders=read(J01/'ORDERS.json'),permutations={o:{f'Q{int(k)+1:02}':v for k,v in d.items()} for o,d in assignments.items()},fixed_before_run=True,same_across_conditions=True,GT_used=False))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in {**p['pins'],**p['inputs']}.items():assert sha(ROOT/f)==h,f
    assert sha(METHOD/'FREEZE_J01.json')==p['freeze_sha256'] and sha(J01/'PREDICTION_BARRIER.json')==p['J01_barrier_sha256']
    return p


def full_temporal_check(model,frames,row,state,layers):
    import torch
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,offset_batch,inserted_state
    from methods.tastvg_dual_evidence_j0_v1.method import combine_layers
    b=make_batch(frames,row['frame_ids'],row['input'],model);outs=[];records=[]
    with torch.no_grad(),query_subject(model,b,row['parses']['subject']),inserted_state(model,state):
        for j in (0,1):
            v=offset_batch(b,j)
            with torch.autocast('cuda',dtype=torch.float16):z=model(v['videos'],v['texts'],v['targets'],iteration_rate=-1)
            outs.append(z);records.append(dict(frame_ids=v['targets'][0]['frame_ids']))
    boxes=[torch.stack([z['pred_boxes'] for z in o['aux_outputs']]+[o['pred_boxes']]).cpu() for o in outs]
    logits=[torch.stack([z['pred_sted'] for z in o['aux_outputs']]+[o['pred_sted']]).cpu() for o in outs]
    actual=combine_layers(boxes,logits,records,row['frame_ids'])
    for a,b in zip(actual,layers):
        assert torch.equal(a['boxes'],b['boxes']) and a['indices']==b['indices']
        assert all(torch.equal(x,y) for x,y in zip(a['logits'],b['logits']))
    return dict(full_pipeline_exact=True,decoder_layers=6,offsets=2)


def run():
    p=verify();prior=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'));tick=time.monotonic();done=updates=0;failure=None;state='failed';lease=None;policy=None
    try:
        import numpy as np,torch
        from methods.decota_final_simplified_v1.tensors import state_hash,detached
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.tastvg_dual_evidence_j0_v1.method import fast_rerank
        from vg_tta.tastvg_matched_ablation_a2_v1 import AblationMethod
        from vg_tta.tastvg_native_spatial_rollout_s05_v1 import reinsert
        from vg_tta.exact_frame_decode_audit_v2 import decode
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs;lease=gpu_lease()
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        def guard(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ['GT_SUBSET','/ROWS.json','/SUMMARY.json']):raise PermissionError('J0 inference forbids GT-derived metrics')
        sys.addaudithook(guard);install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        model=model_load('hcstvg1_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256']
        support=load(NATIVE/'PARAMETER_SUPPORT.pt');deltas=[{n:(v-support['center'][n]).cuda() for n,v in x.items()} for x in support['states']];policy=AblationMethod(model,deltas,p['arms'][0])
        assert all(torch.equal(v.cpu(),support['center'][n]) for n,v in policy.actor.initial.items())
        hb=read(S0/'H_BARRIER.json');eb=read(S0/'EXPERT_BARRIER.json');tb=read(TEMP/'C2_BARRIER.json');ob=read(OLD/'CAPTURE_BARRIER.json');rein=[];temporal_rein=[];streams=[]
        for ablation,order,cond in [(a,o,c) for a in p['arms'] for o in p['orders'] for c in CONDS]:
            policy.arm=ablation
            policy.reset();previous=state_hash(policy.actor.state());assert previous==state_hash(policy.actor.initial);start=previous;local_updates=0
            for arrival,parent in enumerate(p['orders'][order]):
                row=next(r for r in p['rows'] if r['ordinal']==parent)
                assert prior+time.monotonic()-tick<p['cap_seconds']-20 and shutil.disk_usage(ROOT).free>8*2**30
                name=f"{row['ordinal']:03}.pt";hr=f'capture/{cond}/{name}';assert sha(S0/hr)==hb['files'][hr];data=device_tree(load(S0/hr),'cuda');scheduled=arrival in p['expert_indices'];called=[];cached_temporal=[]
                def temporal_provider():
                    assert scheduled;rel=f'c2/{cond}/{name}';assert sha(TEMP/rel)==tb['files'][rel];called.append('temporal');x=load(TEMP/rel);cached_temporal.append(x);return x
                def spatial_provider():
                    assert scheduled;rel=f'expert/{cond}/{name}';assert sha(S0/rel)==eb['files'][rel];called.append('spatial');return load(S0/rel)
                policy.assignment=p['permutations'][order][str(parent)]
                result,post_ev=policy.arrive(data,scheduled,temporal_provider,spatial_provider)
                assert result['pre_state_sha256']==previous and called==(['temporal','spatial'] if scheduled else [])
                result.update(ablation=ablation,order=order,parent=row['ordinal'],condition=cond,arrival=arrival,expert_scheduled=scheduled,pixel_sha256=data['pixel_sha256'],expert_reads=called)
                # Read-only controls; no labels or previous task scores.
                assert sha(OLD/hr)==ob['files'][hr];old=load(OLD/hr)
                assert old['pixel_sha']==data['pixel_sha256']
                assert torch.equal(old['native']['boxes'],data['prediction']['boxes'].cpu())
                if arrival==0:
                    assert torch.equal(result['prediction']['boxes'],old['native']['boxes']) and result['prediction']['indices']==old['native']['indices']
                    assert all(torch.equal(a,b) for a,b in zip(result['prediction']['logits'],old['native']['logits']))
                if result['updated']:updates+=1;local_updates+=1
                fast=old['native']
                if scheduled:
                    teacher=cached_temporal[0];fast,fd=fast_rerank(old['native'],old['candidates']['temporal'],teacher)
                    assert fd['selected']==teacher['selected'] and fd['scores']==teacher['scores'];result['fast_control']=fd
                    result['temporal_support_changed']=result['temporal']['candidates']!=old['candidates']['temporal']
                    result['temporal_selected_interval_changed']=result['output_prediction']['indices']!=fast['indices']
                    if result['displacement_from_source']==0:
                        assert result['temporal']['candidates']==old['candidates']['temporal']
                result['arms']={'Frozen':old['native'],'Fast-only':fast,'Slow-only':result['prediction'],'Final':result['output_prediction']}
                assert torch.equal(result['arms']['Final']['boxes'],result['arms']['Slow-only']['boxes'])
                if not scheduled:assert result['arms']['Final']['indices']==result['arms']['Slow-only']['indices'] and result['arms']['Fast-only']['indices']==old['native']['indices']
                if (order,cond) in [('order1','clean'),('order5','frame_drop_5')] and ((result['updated'] and local_updates==1) or arrival==15):
                    frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,_=observation(row,cond,frames);audit=reinsert(model,shifted,row,policy.actor.state(),post_ev);rein.append(dict(ablation=ablation,order=order,parent=row['ordinal'],condition=cond,arrival=arrival,learned_state=result['displacement_from_source']>0,**audit));del frames,shifted
                if (order,cond) in [('order1','clean'),('order5','frame_drop_5')] and arrival==8:
                    frames,_=decode(row['input']);shifted,_=observation(row,cond,frames);audit=full_temporal_check(model,shifted,row,device_tree(result['pre_state'],'cuda'),result['temporal_layers']);temporal_rein.append(dict(ablation=ablation,order=order,parent=row['ordinal'],condition=cond,learned_state=True,**audit));del frames,shifted
                assert state_hash(policy.actor.state())==result['post_state_sha256'];previous=result['post_state_sha256'];save(OUT/'online'/ablation/order/cond/name,result);done+=1
                status(OUT/'STATUS.json',dict(status='running',done=done,total=960,updates=updates,order=order,condition=cond));print('J01',done,960,ablation,order,cond,arrival,'expert',scheduled,'updated',result['updated'],flush=True)
                del data,result,post_ev,old,cached_temporal;gc.collect();torch.cuda.empty_cache()
            streams.append(dict(ablation=ablation,order=order,condition=cond,start_sha256=start,final_sha256=previous,updates=local_updates));save(OUT/'final_states'/ablation/order/f'{cond}.pt',detached(policy.actor.state(),'cpu'))
        policy.close();policy=None;assert state_hash(model.state_dict())==mh
        verify();assert len(rein)==8 and len(temporal_rein)==4
        write(OUT/'REINSERTION_AUDIT.json',dict(status='pass',spatial=rein,temporal_all_layers=temporal_rein));write(OUT/'STATE_CHAIN.json',dict(status='pass',streams=streams));write(OUT/'PREDICTION_BARRIER.json',dict(cells=960,arms=4,updates=updates,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'online').rglob('*.pt')},final_states={str(f.relative_to(OUT)):sha(f) for f in (OUT/'final_states').rglob('*.pt')},GT_read=False,model_restored=True,time=time.time()));state='completed'
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if policy:policy.close()
        r=dict(status=state,done=done,updates=updates,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',r);status(OUT/'STATUS.json',r)
        if lease:lease.close()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
