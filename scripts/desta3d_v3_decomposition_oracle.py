"""Fixed PANEL16, shared-F native gradients and one equal-budget correction."""
import argparse,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,check_pins,local_dependencies,allocation,seal,tensor_sha
from scripts.desta3d_v3_privileged_ptd_qualification import B1,B1_SHA,ADAPTER_SHA,equal
PROTOCOL=ROOT/'protocols/desta3d_v3_decomposition_oracle_v1.md'
ARMS=['original','T_only','S_only','decomposed','joint','joint_pass_matched']
OLD=OUT/'oracle001'

def register(name):
    import torch,unittest,shutil
    from tests.test_desta3d_v3_decomposition import Contracts
    from vg_tta.desta3d_v3_free_actuation import native_targets
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert result.wasSuccessful() and result.testsRun==6
    d=OUT/name;assert not d.exists() and sha(B1)==B1_SHA
    rows=read(PANEL/'INPUTS.json');labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')}
    assert len(rows)==len({r['source'] for r in rows})==16
    assert read(OLD/'COMPLETE.json')['seal_sha']==sha(OLD/'PREDICTIONS_SEAL.json')
    sealed=read(OLD/'PREDICTIONS_SEAL.json')['files'];baselines=[];support=[]
    for i,row in enumerate(rows):
        p=OLD/'episodes'/f'{i:02}'/'original.pt';assert sha(p)==sealed[str(p.relative_to(OLD))];baselines.append(p)
        pred=torch.load(p,weights_only=False,map_location='cpu');entry={'index':i}
        for b in ['event','spatial']:
            t,v=native_targets(labels[row['key']],pred,b);entry[b]=dict(target_shape=list(t.shape),valid_actions=int(v.sum()))
        support.append(entry)
    assert shutil.disk_usage(ROOT).free>8*2**30+24*2**30
    cfg=dict(stage='source_shared_F_decomposition_oracle',phase_seconds=3600,maximum_new_bytes=24*2**30,
       minimum_free_bytes=8*2**30,cumulative_cap=None,seed=20260927,queries=16,parents=16,arms=ARMS,
       optimizer_steps=0,analytic_steps=1,backward_calls=32,native_predictions=96,finite_fixed_support_reads=128,
       checkpoint_sha=B1_SHA,adapter_sha=ADAPTER_SHA,source_label_sha=sha(PANEL/'SOURCE_RECORDS.json'),
       source_GT_in_worker=True,target_input=False,target_GT=False,ratios={'event':.087687,'spatial':.170316},
       variable='FP32 stock F before frozen B1; q fixed; identity plus reader Jacobian',
       joint='union projection of unit(gT)+unit(gS); norm squared equals sum branch norms squared',
       extra_joint='same J divided sqrt2: equal sum of actual two-pass squared perturbations',
       no_selection=True,baseline_support=support)
    write(d/'CONFIG.json',cfg);write(d/'INPUTS.json',rows)
    write(d/'CPU_PREFLIGHT.json',dict(status='passed',tests=6,source_support=support,real_GPU_untested=True))
    paths=local_dependencies([Path(__file__),PROTOCOL,d/'CONFIG.json',d/'INPUTS.json',d/'CPU_PREFLIGHT.json',B1,
        ROOT/'vg_tta/desta3d_v3_decomposition.py',ROOT/'tests/test_desta3d_v3_decomposition.py',
        PANEL/'SOURCE_RECORDS.json',OLD/'PREDICTIONS_SEAL.json',OLD/'COMPLETE.json',*baselines,
        ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
        ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors'])
    write(d/'LOCK.json',{'pins':{str(p):sha(p) for p in paths}})
    write(d/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',source_training_exposed=True,
        no_best_step=True,GT_oracle_not_TTA=True,phase_seconds=3600,cumulative_cap=None))

def run(name):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';d=OUT/name
    with allocation(d) as (cfg,guard):
        import torch,gc,math
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,inputs_for
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_tta8_recovery_v2 import observe_time
        from scripts.desta3d_v2_p0 import adapter_sha256
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v3_actuation_full_vocab import capture_teacher,replay_branch
        from vg_tta.desta3d_v3_actuation_support import spatial_positions
        from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
        from vg_tta.desta3d_v3_free_actuation import native_targets,native_ce,span_basis
        from vg_tta.desta3d_v3_decomposition import shared_fields,corrections,union_basis,geometry,norm,project
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        from vg_tta.exact_frame_decode_audit_v2 import decode
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        import model.ptd_generation as pg
        a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        a.load_state_dict(torch.load(B1,map_location='cpu',weights_only=False)['adapter']);a.set_train_stage('frozen')
        assert adapter_sha256(a)==ADAPTER_SHA
        qt,rt=span_basis(a.out_proj_event.weight);qs,rs=span_basis(a.out_proj_spatial.weight);qj,sv=union_basis(qt,qs)
        torch.save(dict(T=qt.cpu(),S=qs.cpu(),J=qj.cpu(),singular_values=sv,R_T=rt.cpu(),R_S=rs.cpu(),
                        gate_T=a.gate_event.sigmoid().cpu(),gate_S=a.gate_spatial.sigmoid().cpu()),d/'BASIS.pt')
        labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};count=back=finite=0
        def load(p):return torch.load(p,weights_only=False,map_location='cpu')
        def frozen():
            assert adapter_sha256(a)==ADAPTER_SHA
            assert all(not x.requires_grad and x.grad is None for x in a.parameters())
            assert all(not x.requires_grad and x.grad is None for x in model.parameters())
        for i,row in enumerate(read(d/'INPUTS.json')):
            guard();ep=d/'episodes'/f'{i:02}';ep.mkdir(parents=True)
            frames,ids=decode(row['input']);prompt,pre=inputs_for(row,pr,frames)
            f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            stock=f['visual_grid'].detach().clone();old=load(OLD/'episodes'/f'{i:02}'/'original.pt')
            support={k:tensor_sha(v) for k,v in f.items() if isinstance(v,torch.Tensor)}
            support.update({k:tensor_sha(prompt[k]) for k in ['input_ids','pixel_values_videos','video_grid_thw']})
            assert support==old['support'] and equal(pre,old['preprocess'])
            write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],support=support,preprocess=pre,frame_ids=ids,
                common_F_sha=tensor_sha(stock),query_features_frozen=True,source_GT_for_oracle=True,target_read=False))
            with shared_fields(a,{'event':stock,'spatial':stock},['event','spatial']):
                (native,trace),td=observe_time(pg,pr,len(ids),lambda:capture_teacher(model,pr,prompt,a,f))
            def prediction(result,td,arm):
                p=prediction_record(result,row,pre,ADAPTER_SHA)
                p.update(arm=arm,readout=details(result),time_distribution=td,support=support,
                    source_GT_oracle=True,decoder_GT_prefix=False,target_read=False,optimizer_steps=0)
                torch.save(p,ep/(arm+'.pt'));validate_prediction(p,len(ids))
                return p
            base=prediction(native,td,'original');count+=1;torch.save(trace,ep/'BASE_TRACE.pt')
            checks={k:equal(base[k],old[k]) for k in ['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval',
                'format_ok','preprocess','adapter_sha','event_completion','spatial_completion','readout','time_distribution']}
            write(ep/'BASELINE_REPLAY.json',checks);assert all(checks.values()),checks
            endpoints={b:native[b+'_injection']['updated_tokens'].detach().cpu().clone() for b in ['event','spatial']}
            del native
            gradients={};objectives={};lab=labels[row['key']]
            for b,j,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
                guard();release_free_host_arenas();torch.cuda.empty_cache()
                target_support={**base,'positions':spatial_positions(trace)} if b=='spatial' else base
                target,valid=native_targets(lab,target_support,b)
                if b=='spatial':target=torch.tensor(trace['coordinate_ids'])[target]
                expected=trace['branches'][j]['logits'][kind]
                leaf=stock.detach().clone().requires_grad_()
                with shared_fields(a,{b:leaf},[b]) as calls:
                    logits,audit=replay_branch(model,prompt,a,f,trace,b)
                    exact=torch.equal(logits.detach().cpu(),expected)
                    torch.save(dict(native=expected,replay=logits.detach().cpu(),exact=exact,targets=target,valid=valid,
                        F_sha=tensor_sha(leaf),cache=audit),ep/(b+'_INITIAL_FORWARD.pt'))
                    assert exact,'Initial differentiable shared-F replay differs from native'
                    loss=native_ce(logits,target.cuda(),valid.cuda());loss.backward();back+=1
                assert leaf.grad is not None and torch.isfinite(leaf.grad).all() and norm(leaf.grad)>0
                gradients[b]=leaf.grad.detach().clone();objectives[b]=dict(targets=target,valid=valid,CE=float(loss.detach()),kind=kind)
                frozen();print('DECOMPOSITION_GRADIENT',i+1,16,b,float(loss.detach()),norm(gradients[b]),flush=True)
                del leaf,logits,loss,audit;gc.collect();torch.cuda.empty_cache()
            deltas=corrections(gradients['event'],gradients['spatial'],qt,qs,qj,stock,cfg['ratios'])
            geo=geometry(gradients['event'],gradients['spatial'],deltas)
            geo['projection_fraction']={b:norm(project(gradients[b],q))/norm(gradients[b]) for b,q in [('event',qt),('spatial',qs)]}
            geo['stock_norm']=norm(stock);geo['projected_gradient_cosine']=geometry(project(gradients['event'],qt),project(gradients['spatial'],qs),deltas)['gradient_cosine']
            torch.save(dict(stock=stock.cpu(),gradients={b:g.cpu() for b,g in gradients.items()},deltas={k:v.cpu() for k,v in deltas.items()},
                objectives=objectives,geometry=geo),ep/'GEOMETRY.pt')
            write(ep/'GEOMETRY.json',geo)
            # These forwards keep original native reference/interval/anchors.
            for kind,delta in deltas.items():
                for b in ['event','spatial']:
                    guard();release_free_host_arenas();torch.cuda.empty_cache()
                    with torch.no_grad(),shared_fields(a,{b:stock+delta},[b]):
                        logits,audit=replay_branch(model,prompt,a,f,trace,b)
                        ob=objectives[b];ce=native_ce(logits,ob['targets'].cuda(),ob['valid'].cuda())
                    torch.save(dict(logits=logits.cpu(),CE=float(ce),targets=ob['targets'],valid=ob['valid'],cache=audit,
                        baseline_trace_sha=sha(ep/'BASE_TRACE.pt'),F_sha=tensor_sha(stock+delta)),ep/f'FINITE_{kind}_{b}.pt')
                    finite+=1;del logits,ce,audit;gc.collect()
            plans={'T_only':('T',None),'S_only':(None,'S'),'decomposed':('T','S'),'joint':('J','J'),'joint_pass_matched':('J_pass','J_pass')}
            for arm,(dt,ds) in plans.items():
                guard();values={b:stock+(deltas[k] if k else 0) for b,k in [('event',dt),('spatial',ds)]}
                expected_endpoints={}
                with torch.no_grad():
                    for b in ['event','spatial']:
                        rr=a(values[b],f['query_tokens'],query_mask=f['query_mask'],frame_times=f['frame_times'])
                        expected_endpoints[b]=rr['updated_tokens_'+b].reshape_as(endpoints[b]).to(torch.bfloat16).cpu()
                    del rr
                with shared_fields(a,values,['event','spatial'],allow_prefix=True) as calls:
                    native,td=observe_time(pg,pr,len(ids),lambda:decode_shared_reference_two_pass(model,pr,prompt,a,f))
                p=prediction(native,td,arm);count+=1
                if len(calls)!=2:assert not native['event']['format_ok'] and native['spatial'] is None
                injection={}
                for b,k in [('event',dt),('spatial',ds)]:
                    state=native.get(b+'_injection')
                    if state is None:injection[b]={'missing':'upstream event format failure'};continue
                    actual=state['updated_tokens'].detach().cpu()
                    assert torch.equal(actual,expected_endpoints[b])
                    diff=actual.float()-endpoints[b].float()
                    injection[b]=dict(actual_sha=tensor_sha(actual),baseline_sha=tensor_sha(endpoints[b]),
                        new_F_sha=tensor_sha(values[b]),delta_norm=0. if k is None else norm(deltas[k]),
                        realized_FP32_F_delta_norm=norm(values[b]-stock),postcast_delta_norm=norm(diff),
                        changed_elements=int((actual!=endpoints[b]).sum()),elements=actual.numel(),exact_expected=True)
                write(ep/(arm+'_INJECTION.json'),injection)
                if arm=='S_only':assert equal(p['time_distribution'],base['time_distribution']) and p['event_completion']==base['event_completion']
                frozen();print('DECOMPOSITION_NATIVE',i+1,16,arm,p['format_ok'],flush=True)
                del native,p,values,expected_endpoints,diff;gc.collect();torch.cuda.empty_cache()
            assert tensor_sha(stock)==support['visual_grid'];frozen()
            write(ep/'COMPLETE.json',dict(key=row['key'],predictions=6,backward_calls=2,finite_reads=8,optimizer_steps=0,
                exact_state=True,adapter_sha=ADAPTER_SHA,common_F_sha=tensor_sha(stock)))
            del f,stock,prompt,old,base,trace,endpoints,gradients,objectives,deltas,delta,target,valid,expected;gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in d.rglob('*') if p.is_file())<cfg['maximum_new_bytes']
        write(d/'COMPLETE.json',dict(status='all_fixed_predictions_sealed_not_scored',queries=16,parents=16,predictions=count,
            backward_calls=back,finite_reads=finite,optimizer_steps=0,seal_sha=seal(d,count),target_read=False,
            GPU_peak_allocated=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True)
    args=p.parse_args();globals()[args.action](args.name)
