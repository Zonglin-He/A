"""Fixed-source-checkpoint reference-contract audit. No training or target GT.

Separate registrations preserve the 4-query engineering check and the full
198-query paired readout. Source labels are read by a later CPU scorer only.
"""
from __future__ import annotations
import argparse,fcntl,gc,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.desta3d_v2_p0 import adapter_sha256,sha,read,write
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,cpu_copy,tensor_sha256
PARENT=ROOT/'artifacts/desta3d_v2';BASE=PARENT/'shared_reference_v1'

def register(run_id,scope):
    dest=BASE/run_id;assert not dest.exists()
    meta=read(BASE/'CHECKPOINT.json');assert sha(Path(meta['checkpoint']))==meta['sha256']
    allrows=sorted([r for r in read(PARENT/'source_fit/INPUTS.json') if r['split']=='validation'],key=lambda r:r['key'])
    if scope=='pilot':
        rows=[];seen=set()
        for r in allrows:
            if r['source'] not in seen:rows.append(r);seen.add(r['source'])
            if len(rows)==4:break
    else:
        assert read(BASE/'pilot003/COMPLETE.json')['interface_pass']
        rows=allrows
    trainrow=next(r for r in sorted(read(PARENT/'source_fit/INPUTS.json'),key=lambda r:r['key']) if r['split']=='train')
    config={'run_id':run_id,'scope':scope,'seed':20260927,'queries':len(rows),'parents':len({r['source'] for r in rows}),
      'checkpoint':meta,'arms':['independent_reference','shared_reference_time'],'revision':'cached official staged prefix, after preserved pilot001 whole-prefix failure',
      'sampling':'lexical source-val keys; pilot first unique4parents, full all198queries/31parents',
      'source_training_updates':0,'source_GT_used':'one fixed train record for task-vs-aux gradient decomposition only, no optimizer','target_GT_read':False,'source_val_GT_read':False,
      'event_pair':'same adapter/pixels/time and deterministic event output; assert token-completion equality',
      'zero_control':'pilot same four queries old/new zero gates; compare full interval/reference/boxes; no task-score acceptance gate',
      'gradient_probe':'pilot lexical first source-train query, clean teacher/gamma0.9 student, one ephemeral AdamW1e-5 wd0 calibration step; no labels; reset afterwards',
      'gradient_train_key':trainrow['key'],'TTA_parameter_count':66816,'TTA_groups':['branch_film','norm_affine'],
      'gates':'frozen because declared objective is pre-gate','TTA_losses':'latent normalized MSE + referent Bernoulli KL + event Bernoulli KL + mean parameter anchor; alignment=0 joint=0',
      'residual_distribution_control':'same fixed event reference/time, spatial residual on vs zero; KL on 1001-coordinate vocabulary', 'state_selection':False,'phase_seconds':1800,'cumulative_cap_seconds':None,'free_disk_bytes':8*2**30,
      'scoring':'after all pairs seal, source-only full v/s/t, parent paired CI and negative tails, never choose checkpoint by this comparison'}
    write(dest/'CONFIG.json',config);write(dest/'INPUTS.json',rows);write(dest/'GRADIENT_INPUT.json',trainrow)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_shared_reference.py',ROOT/'vg_tta/desta3d_v2_shared_reference_cached.py',ROOT/'vg_tta/desta3d_v2_tta_objective.py',ROOT/'scripts/desta3d_v2_source_gradient_audit.py',
       ROOT/'vg_tta/desta3d_v2.py',ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',
       ROOT/'scripts/desta3d_v2_source_fit.py',ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
       PARENT/'source_fit/LOCK.json',BASE/'CHECKPOINT.json',Path(meta['checkpoint']),dest/'CONFIG.json',dest/'INPUTS.json',dest/'GRADIENT_INPUT.json',ROOT/'methods/CURRENT_METHOD.json']
    if scope=='full':paths.extend([BASE/'pilot003/COMPLETE.json',BASE/'pilot003/PREDICTIONS_SEAL.json'])
    write(dest/'LOCK.json',{'pins':{str(p):sha(p) for p in paths},'original_source_lock_preserved':True})
    print('REGISTERED',dest,flush=True)

def save_pt(path,payload):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.pt');torch.save(cpu_copy(payload),tmp);os.replace(tmp,path)

def details(result):
    sp=result.get('spatial') or {};e=result['event']
    old_semantic=sp.get('semantic',[])
    spatial_ref=sp.get('shared_reference_token_ids',[int(t) for block,_done in old_semantic for t in block])
    out={'event_completion':e['completion'],'spatial_reference_token_ids':spatial_ref,
        'spatial_completion':sp.get('completion'),'spatial_prefix_length':sp.get('spatial_prefix_length'),
        'cache_policy':result.get('cache_policy'),'coordinate_logits':sp.get('logits'),
        'raw_blocks':sp.get('raw_box_blocks',sp.get('raw_blocks'))}
    for n in ('event_injection','spatial_injection'):
        x=result[n]
        out[n]=None if x is None else {k:x[k] for k in ('calls','zero_exact','changed_elements','relative_injection_norm')}
        if x is not None:out[n]['elements']=x['updated_tokens'].numel()
    return out

def gradient_probe(model,pr,adapter,run):
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for,inputs_for
    from vg_tta.desta3d_v2_ptd import capture_stock_fields
    from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass,_spatial_decode_official_cached as _spatial_boxes_from_shared_prefix
    from vg_tta.desta3d_v2_tta_objective import configure_pre_gate_calibration,calibration_objective,gradient_groups
    import model.ptd_generation as pg
    row=read(run/'GRADIENT_INPUT.json');assert row['split']=='train'
    frames,ids=frames_for(row,'clean');prompt,prep=inputs_for(row,pr,frames)
    fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
    aug,ids2=frames_for(row,'clean','gamma0.9');view,augprep=inputs_for(row,pr,aug)
    fields2=capture_stock_fields(model,pr,view,row['input']['caption'],ids2,row['input']['fps'])
    assert ids==ids2 and torch.equal(fields['frame_times'],fields2['frame_times'])
    assert fields['visual_grid'].shape==fields2['visual_grid'].shape
    initial=cpu_copy(adapter.state_dict());initial_hash=adapter_sha256(adapter)
    setup=configure_pre_gate_calibration(adapter);assert setup['parameter_count']==66816
    def forward(f):return adapter(f['visual_grid'],f['query_tokens'],query_mask=f['query_mask'],frame_times=f['frame_times'])
    with torch.no_grad():teacher=forward(fields)
    before=decode_shared_reference_two_pass(model,pr,view,adapter,fields2)
    assert before['event']['format_ok'] and before['spatial'] is not None,'gradient probe requires an actual spatial readout'
    shared=before['event']['shared_reference_time']
    token_ids=pg.build_ptd_token_ids(pr.tokenizer,max_time_tokens=int(view['video_grid_thw'][0,0]))
    student=forward(fields2);loss,terms=calibration_objective(adapter,student,teacher,setup['initial_parameters'])
    adapter.zero_grad(set_to_none=True);loss.backward();grad=gradient_groups(adapter)
    assert grad['branch_film']['nonzero_tensors'] and grad['norm_affine']['nonzero_tensors']
    assert grad['gates']['none_tensors']==2 and all(p.grad is None for p in model.parameters())
    opt=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.)
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],1.));opt.step()
    after,inj=_spatial_boxes_from_shared_prefix(model,pr,view,fields2,adapter,shared,token_ids)
    updated=adapter.state_dict();changed={n:bool(not torch.equal(v.cpu(),initial[n])) for n,v in updated.items()}
    assert not changed['gate_event'] and not changed['gate_spatial']
    assert all(not flag or n in setup['initial_parameters'] for n,flag in changed.items())
    b=before['spatial']['logits'].float();a=after['logits'].float();assert a.shape==b.shape
    kl=(b.log_softmax(-1).exp()*(b.log_softmax(-1)-a.log_softmax(-1))).sum(-1).mean()
    report={'key':row['key'],'source':row['source'],'GT_read':False,'optimizer_steps':1,'ephemeral_source_interface_only':True,
      'initial_adapter_sha':initial_hash,'updated_parameter_count':setup['parameter_count'],'gradients':grad,
      'loss':float(loss.detach()),'terms':{k:float(v.detach()) for k,v in terms.items()},'clip_norm_before':norm,
      'changed_parameter_tensors':sum(changed.values()),'gates_unchanged':True,'backbone_grad_none':True,
      'fixed_event_reference_time_for_after':True,'coordinate_logits_shape':list(a.shape),
      'logit_changed_count':int((a!=b).sum()),'logit_max_abs_change':float((a-b).abs().max()),
      'coordinate_KL_before_to_after':float(kl),'final_boxes_changed':not torch.equal(before['spatial']['boxes'],after['boxes']),
      'cast_residual_before':details(before)['spatial_injection'],
      'cast_residual_after':{k:inj[k] for k in ('changed_elements','relative_injection_norm')},
      'preprocess_clean':prep,'preprocess_aug':augprep,'scope':'actual frozen PTD output, no task score or generalization claim'}
    save_pt(run/'GRADIENT_PROBE_OUTPUTS.pt',{'before':b,'after':a,'report':report,'parameter_changed':changed})
    adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True);adapter.eval()
    assert adapter_sha256(adapter)==initial_hash
    report['adapter_exactly_reset']=True;write(run/'REAL_TTA_GRADIENT.json',report)
    from scripts.desta3d_v2_source_gradient_audit import audit_source_gradient
    audit_source_gradient(model,pr,adapter,row,fields,run)
    return report

def run(run_id):
    dest=BASE/run_id;c=read(dest/'CONFIG.json')
    for p,h in read(dest/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    for p,h in read(PARENT/'source_fit/LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    assert not (dest/'STARTED.json').exists(),'register new invocation after a preserved failure'
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();status='running';failure=None;prior=sum(scan_nested_gpu_receipts(p)[0] for p in [ROOT/'artifacts/desta3d_v1',PARENT])
    def guard():
        assert shutil.disk_usage(ROOT).free>=c['free_disk_bytes'],'disk reserve'
        if time.monotonic()-began>c['phase_seconds']:raise RuntimeError('engineering phase review required; partials preserved')
    try:
        guard();write(dest/'STARTED.json',{'pid':os.getpid(),'time':time.time(),'prior_seconds':prior})
        torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,decode_two_pass
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass,_spatial_decode_official_cached
        from scripts.desta3d_v2_source_fit import prediction_record
        pr=processor_load()
        import model.ptd_generation as pg
        model=model_load();adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        ck=torch.load(c['checkpoint']['checkpoint'],map_location='cpu',weights_only=False);adapter.load_state_dict(ck['adapter'])
        ahash=adapter_sha256(adapter);assert ahash==c['checkpoint']['adapter_sha256'];checks=[]
        for i,row in enumerate(read(dest/'INPUTS.json')):
            guard();frames,ids=frames_for(row,'clean');prompt,prep=inputs_for(row,pr,frames)
            fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            pair={};diag={}
            for arm,fn in [('independent_reference',decode_two_pass),('shared_reference_time',decode_shared_reference_two_pass)]:
                result=fn(model,pr,prompt,adapter,fields)
                pair[arm]=prediction_record(result,row,prep,ahash);diag[arm]=details(result)
                if arm=='shared_reference_time' and result.get('spatial') and 'logits' in result['spatial']:
                    fixed=result['event']['shared_reference_time']
                    tids=pg.build_ptd_token_ids(pr.tokenizer,max_time_tokens=int(prompt['video_grid_thw'][0,0]))
                    off,_=_spatial_decode_official_cached(model,pr,prompt,fields,adapter,fixed,tids,gate_override=0.)
                    if 'logits' in off:
                        a=result['spatial']['logits'].float();b=off['logits'].float();assert a.shape==b.shape
                        kl=(a.log_softmax(-1).exp()*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).mean()
                        diag[arm]['same_prefix_residual_control']={'without_logits':b,'KL_with_to_without':float(kl),
                          'logit_changed':int((a!=b).sum()),'max_abs_change':float((a-b).abs().max()),
                          'format_without':off['format_ok'],'fixed_reference_time':True}
                    else:diag[arm]['same_prefix_residual_control']={'unavailable':'zero residual format did not expose logits'}
                    del off
                save_pt(dest/'predictions'/arm/f'{i:03}.pt',pair[arm])
                del result
            assert pair['independent_reference']['event_completion']==pair['shared_reference_time']['event_completion']
            assert pair['independent_reference']['interval']==pair['shared_reference_time']['interval']
            same_ref=diag['independent_reference']['spatial_reference_token_ids']==diag['shared_reference_time']['spatial_reference_token_ids']
            check={'key':row['key'],'source':row['source'],'event_completion_equal':True,'interval_equal':True,
              'references_equal':same_ref,'format_ok':{a:r['format_ok'] for a,r in pair.items()}}
            if c['scope']=='pilot':
                zero_old=decode_two_pass(model,pr,prompt,adapter,fields,gate_override=0.)
                zero_new=decode_shared_reference_two_pass(model,pr,prompt,adapter,fields,gate_override=0.)
                check['zero_interval_equal']=zero_old['interval']==zero_new['interval']
                check['zero_reference_equal']=details(zero_old)['spatial_reference_token_ids']==details(zero_new)['spatial_reference_token_ids']
                check['zero_formats']=[zero_old['format_ok'],zero_new['format_ok']]
                check['zero_boxes_equal']=torch.equal((zero_old.get('spatial') or {}).get('boxes',torch.empty(0,4)),(zero_new.get('spatial') or {}).get('boxes',torch.empty(0,4)))
                diag['zero']={'old':prediction_record(zero_old,row,prep,ahash),'new':prediction_record(zero_new,row,prep,ahash),'check':check}
                del zero_old,zero_new
            save_pt(dest/'diagnostics'/f'{i:03}.pt',diag);checks.append(check);write(dest/'checks'/f'{i:03}.json',check)
            print('PAIR',i,row['key'],check,flush=True)
            del fields,prompt,frames,pair,diag;gc.collect()
        assert adapter_sha256(adapter)==ahash
        if c['scope']=='pilot':
            interface=all(r['zero_interval_equal'] and r['zero_reference_equal'] and r['zero_boxes_equal'] and all(r['zero_formats']) for r in checks)
            if not interface:raise RuntimeError('zero-gate shared-prefix control differs; inspect before expanding')
            gradient_probe(model,pr,adapter,dest)
        pins={str(p):sha(p) for folder in ('predictions','diagnostics') for p in sorted((dest/folder).rglob('*.pt'))}
        write(dest/'PREDICTIONS_SEAL.json',{'pins':pins,'source_val_GT_read':False,'target_GT_read':False,'queries':len(checks)})
        write(dest/'COMPLETE.json',{'status':'completed','interface_pass':True,'checks':checks,'source_val_GT_read':False,'target_GT_read':False})
        status='completed'
    except BaseException:
        status='failed';failure=traceback.format_exc();write(dest/'FAILURE.json',{'failure':failure,'time':time.time()});raise
    finally:
        seconds=time.monotonic()-began
        write(PARENT/'receipts'/f'shared_reference_{run_id}.json',{'status':status,'seconds':seconds,'prior_seconds':prior,
          'cumulative_seconds':prior+seconds,'failure':failure,'cumulative_cap_seconds':None,
          'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        lease.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['register','run']);a.add_argument('--run-id',required=True)
    a.add_argument('--scope',choices=['pilot','full'],default='pilot');x=a.parse_args()
    register(x.run_id,x.scope) if x.action=='register' else run(x.run_id)
