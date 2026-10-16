"""Finite two-stage latent query oracles, then conditional negative-critic audit."""
import argparse, gc, os, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
OUT=ROOT/'artifacts/spatial_query_negative_oracle_v1'
PARENT=ROOT/'artifacts/actionness_spatial_attribution_v1'
CACHE=ROOT/'artifacts/spatial_support_source_p0_v1/target_cache/correct'
TEACHERS=ROOT/'artifacts/decota_referential_control_v1/teachers'
COHORTS=['hcstvg1_test','vidstg_test']
CODE=['scripts/run_spatial_query_negative_oracle_v1.py','vg_tta/spatial_query_negative_oracle_v1.py',
      'protocols/spatial_query_negative_oracle_v1.md','tests/test_spatial_query_negative_oracle_v1.py']


def prepare():
    from vg_tta.spatial_query_negative_oracle_v1 import proposals
    if (OUT/'LOCK.json').exists():return verify()
    p=read(PARENT/'LOCK.json');rows=[]
    for r in p['rows']:
        stem=r['key'].replace(':','_');c=CACHE/f'{stem}.pt';t=TEACHERS/f'{stem}.pt'
        assert sha(r['path'])==r['sha256']
        for f in (c,t):assert sha(f)==read(f.with_suffix('.json'))['sha256'],f
        pools=proposals(load(t));dest=OUT/'proposals'/f'{stem}.pt'
        save(dest,dict(key=r['key'],pool=pools,GT_access=False,teacher_sha256=sha(t)))
        rows.append({**r,'cache':str(c),'cache_sha256':sha(c),'teacher':str(t),'teacher_sha256':sha(t),
                     'proposals':str(dest),'proposals_sha256':sha(dest)})
    p=dict(rows=rows,production_pins=p['production_pins'],labels=p['labels'],labels_sha256=p['labels_sha256'],
       code={f:sha(ROOT/f) for f in CODE},parent_lock_sha256=sha(PARENT/'LOCK.json'),
       created=time.time(),historically_exposed=True,model='TA-STVG',parameters=256,
       latent_optimization=True,backbone_weight_updates=False,temporal_updates=False,
       lrs=[.005,.05,.5],primary_lr=.05,gamma=1e-4,steps=[1,5],negative_iou_threshold=.05,
       tau=1.,optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-4,weight_decay=0),
       random_seeds=[20260918,20260919,20260920],max_gpu_seconds=5400,
       bootstrap_replicates=10000,bootstrap_seed=20260914,auto_promote=False)
    write(OUT/'LOCK.json',p);print('LOCKED',len(rows),'exposed sources; proposals sealed before oracle labels',flush=True)
    return p


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in {**p['production_pins'],**p['code']}.items():assert sha(ROOT/f)==h,f
    assert sha(PARENT/'LOCK.json')==p['parent_lock_sha256']
    assert sha(p['labels'])==p['labels_sha256']
    return p


def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),lock_sha256=sha(OUT/'LOCK.json'),created=time.time()))


def exists(f):
    if not f.with_suffix('.json').exists():return False
    r=read(f.with_suffix('.json'))
    assert sha(f)==r['sha256'] and r['lock_sha256']==sha(OUT/'LOCK.json')
    return True


def pause():
    base=ROOT/'artifacts/decota_paper_execution_20260917'
    s=read(base/'background_last/STATUS.json')
    pid=s.get('pid');proc=Path(f'/proc/{pid}/cmdline')
    if s.get('status')!='running' or not proc.exists():return None
    assert 'scripts/run_decota_paper_background_last_20260917.py' in proc.read_bytes().replace(b'\0',b' ').decode()
    workers=[base/'vitta/hc2_to_vid/full/STATUS.json',base/'vid_last/native/full/STATUS.json',
             base/'vid_last/safety/full/STATUS.json']
    snap=dict(coordinator=s,workers={str(f):read(f) for f in workers if f.exists()},time=time.time())
    write(OUT/f'BASELINE_PAUSE_{int(time.time())}.json',snap);os.kill(pid,signal.SIGTERM)
    for _ in range(160):
        if not proc.exists() and not Path(f'/proc/{s["child_pid"]}').exists():break
        time.sleep(.25)
    else:raise RuntimeError('Existing finite coordinator did not stop cleanly')
    return snap


def episode(model,row,p,phase,full_check=False):
    import numpy as np,torch
    from vg_tta.spatial_query_negative_oracle_v1 import (
        QueryReplay,negative_masks,shuffled_masks,candidate_log_probs,complementary)
    from vg_tta.spatial_ssl_diagnostic_v1 import BoxLoss,gradient_vector,cosine
    from methods.decota_final_simplified_v1.tensors import detached
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.backbone import query_subject,make_batch,full_prediction
    start=time.time();assert sha(row['cache'])==row['cache_sha256']
    assert sha(row['proposals'])==row['proposals_sha256'];c=load(row['cache']);pool=load(row['proposals'])['pool']
    s=QueryReplay(model,detached(c['views'],'cuda'),len(c['frame_ids']))
    assert torch.equal(s.zero['boxes'].cpu(),c['zero']['boxes'])
    assert all(torch.equal(a.cpu(),b) for a,b in zip(s.zero['logits'],c['zero']['logits']))
    assert not any(v.requires_grad for v in s.decoder.parameters())
    native=prediction(s.zero['logits'],s.zero['boxes'],c['records'],c['frame_ids'])
    grids=[v['info']['fea_map_size'] for v in c['views']]
    gt=read(p['labels'])[row['key']];truth=torch.tensor(gt['boxes'],dtype=torch.float32,device='cuda')
    flags,diagnostics=negative_masks(pool,truth.cpu(),gt['valid'])
    allpos=np.flatnonzero(gt['valid']).tolist();matched=sorted(flags)
    gta=BoxLoss(truth[allpos],allpos,len(allpos))
    gtm=BoxLoss(truth[matched],matched,len(matched)) if matched else None
    def forward_loss(arm):
        if arm.startswith('G_'):
            vals=s.values(); data=(gta if arm=='G_all' else gtm)(vals['boxes']);extra={}
        else:
            vals,att=s.values_with_attention();lp=candidate_log_probs(att,pool,grids)
            masks=flags if arm=='N' else shuffled_masks(flags,p['random_seeds'][int(arm[1:])])
            data=complementary(lp,masks)
            extra=dict(candidate_probabilities={k:v.detach().exp().cpu() for k,v in lp.items()})
        reg=p['gamma']*s.delta.square().sum()
        return vals,data+reg,data,extra
    s.restore(s.initial)
    grad_gt=gradient_vector(s,gta(s.values()['boxes'])).cpu()
    assert grad_gt.numel()==256 and torch.isfinite(grad_gt).all()
    arms=['G_all','G_match'] if phase=='actuator' else ['N','R0','R1','R2']
    runs={};audits=dict(zero_exact=True,parameters=256,all_network_parameters_frozen=True,
                       temporal_logits_exact=True,full_reinsertion=[],finite_difference=None,lr0=True)
    frames=batch=None;parent=None
    if full_check:
        from vg_tta.exact_frame_decode_audit_v2 import decode
        parent=load(row['path']);frames,ids=decode(c['input']);assert ids==c['frame_ids']
        batch=make_batch(frames,ids,c['input'],model)
    for arm in arms:
        available=arm=='G_all' or bool(matched)
        for lr in p['lrs']:
            name=f'{arm}_lr{lr:g}';s.restore(s.initial)
            opt=torch.optim.AdamW([s.delta],lr=lr,betas=(.9,.999),eps=1e-4,weight_decay=0)
            states={};trace=[];grad0=None;initial_loss=None
            for step in range(6):
                if available:
                    vals,loss,data,extra=forward_loss(arm)
                    entry=dict(step=step,loss=float(loss.detach()),data_loss=float(data.detach()),
                               delta_norm=float(s.delta.detach().norm()),**extra)
                else:
                    vals=s.values();loss=None;entry=dict(step=step,loss=None,data_loss=None,delta_norm=0)
                if step==0:initial_loss=entry['loss']
                trace.append(entry)
                if step in p['steps']:
                    pred=prediction(vals['logits'],vals['boxes'],c['records'],c['frame_ids'])
                    assert pred['indices']==native['indices']
                    assert all(torch.equal(a,b) for a,b in zip(pred['logits'],native['logits']))
                    states[step]=dict(prediction=pred,state=detached(s.state(),'cpu'),
                        box_max_abs=float((pred['boxes']-native['boxes']).abs().max()))
                    if full_check and lr==p['primary_lr'] and arm in ('G_all','G_match','N','R0'):
                        with query_subject(model,batch,parent['parses']['subject']):
                            actual=full_prediction(model,batch,c['frame_ids'],c['records'],s.state(),pred)
                        assert actual['indices']==pred['indices']
                        audits['full_reinsertion'].append(dict(arm=arm,step=step,exact=True))
                if step==5:break
                if available:
                    opt.zero_grad();loss.backward()
                    g=s.delta.grad.detach().clone();assert torch.isfinite(g).all()
                    if step==0:
                        grad0=g.double().cpu()
                        if lr==p['primary_lr']:
                            # A real backward/step with lr=0 must preserve the latent and outputs.
                            zero=torch.optim.AdamW([s.delta],lr=0,eps=1e-4,weight_decay=0)
                            zero.step();assert torch.equal(s.delta,s.initial['spatial.query_residual'])
                            assert torch.equal(s.values()['boxes'],s.zero['boxes'])
                        if phase=='negative' and arm=='N' and lr==p['primary_lr'] and full_check:
                            direction=g.double()/g.double().norm().clamp_min(1e-30)
                            analytic=float((g.double()*direction).sum());h=.01
                            values=[]
                            for sign in (1,-1):
                                with torch.no_grad():s.delta.copy_((sign*h*direction).float())
                                with torch.no_grad():values.append(float(forward_loss(arm)[1]))
                            numerical=(values[0]-values[1])/(2*h)
                            with torch.no_grad():s.delta.zero_()
                            # Restore exact pre-FD gradient for the actual update.
                            s.delta.grad=g.clone()
                            passed=abs(analytic-numerical)<=max(1e-5,.05*abs(analytic))
                            audits['finite_difference']=dict(analytic=analytic,numerical=numerical,h=h,passed=passed)
                            assert passed,audits['finite_difference']
                    trace[-1]['grad_norm']=float(g.norm());opt.step()
                del vals,loss
            runs[name]=dict(arm=arm,lr=lr,available=available,states=states,trace=trace,
                gradient=grad0,gradient_cosine_gt=cosine(grad0,grad_gt) if grad0 is not None else None,
                initial_loss=initial_loss)
    s.restore(s.initial)
    assert torch.equal(s.values()['boxes'],s.zero['boxes'])
    assert not any(v.grad is not None for v in model.parameters())
    return dict(key=row['key'],cohort=row['cohort'],source=row['source'],phase=phase,
        baseline=native,runs=runs,gradient_gt=grad_gt,frame_ids=c['frame_ids'],
        input=c['input'],candidate_diagnostics=diagnostics,negative_flags=flags,
        eligible_positions=matched,all_GT_positions=allpos,GT_access=True,oracle=True,
        audits=audits,seconds=time.time()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())


def analyze(phase,p):
    import numpy as np
    from scripts.analyze_spatial10_components_v1 import aggregate,checked_score
    target=OUT/'analysis'/phase
    if (target/'DECISION.json').exists():return read(target/'DECISION.json')
    labels=read(p['labels']);rows=[]
    for r in p['rows']:
        f=OUT/phase/(r['key'].replace(':','_')+'.pt');assert exists(f);z=load(f);gt=labels[r['key']]
        idx=z['baseline']['indices'];ids=z['frame_ids'];valid=np.array(gt['valid'],bool)
        mask=np.zeros(len(ids),bool);mask[z['eligible_positions']]=True
        def metrics(pred):
            m,q=checked_score(pred['boxes'],gt,ids,idx)
            m['observed_sIoU']=float(q[valid&mask].mean()) if (valid&mask).any() else None
            m['unobserved_sIoU']=float(q[valid&~mask].mean()) if (valid&~mask).any() else None
            return m
        values={'Frozen':metrics(z['baseline'])};diagnostics={}
        for name,run in z['runs'].items():
            for step,s in run['states'].items():values[name+f'_s{step}']=metrics(s['prediction'])
            diagnostics[name]=dict(available=run['available'],cosine_gt=run['gradient_cosine_gt'],
                loss_delta=(run['trace'][-1]['loss']-run['trace'][0]['loss']) if run['available'] else None,
                delta_norm=run['trace'][-1]['delta_norm'])
        rows.append(dict(key=r['key'],source=r['source'],cohort=r['cohort'],values=values,
                    diagnostics=diagnostics,candidate_diagnostics=z['candidate_diagnostics'],
                    eligible_positions=z['eligible_positions'],seconds=z['seconds']))
    summary={};keys=list(rows[0]['values']);metrics=['vIoU_corrected','sIoU','tIoU','observed_sIoU','unobserved_sIoU']
    for c in COHORTS:
        rr=[r for r in rows if r['cohort']==c];src=[r['source'] for r in rr]
        assert len(rr)==len(set(src))==32
        arms={a:{m:aggregate([r['values'][a].get(m) for r in rr],src) for m in metrics} for a in keys}
        contrasts={a:{m:aggregate([r['values'][a][m]-r['values']['Frozen'][m]
              if r['values'][a][m] is not None and r['values']['Frozen'][m] is not None else None for r in rr],src)
              for m in metrics} for a in keys if a!='Frozen'}
        if phase=='negative':
            for lr in p['lrs']:
                for step in (1,5):
                    a=f'N_lr{lr:g}_s{step}';name=a+' - random_mean'
                    contrasts[name]={m:aggregate([r['values'][a][m]-np.mean([r['values'][f'R{i}_lr{lr:g}_s{step}'][m] for i in range(3)])
                        if r['values'][a][m] is not None else None for r in rr],src) for m in metrics}
        summary[c]=dict(queries=32,sources=32,eligible_queries=sum(bool(r['eligible_positions']) for r in rr),
            arms=arms,contrasts=contrasts)
    if phase=='actuator':
        passing=[lr for lr in p['lrs'] if all(summary[c]['contrasts'][f'G_all_lr{lr:g}_s5']['sIoU']['mean']>.001 for c in COHORTS)]
        decision=dict(gate_passed=bool(passing),common_LRs_with_headroom=passing,
                      next='negative_only_oracle' if passing else 'stop_at_actuator')
    else:
        name=f'N_lr{p["primary_lr"]:g}_s5'
        passed=all(summary[c]['contrasts'][name]['sIoU']['mean']>.001 and
           summary[c]['contrasts'][name+' - random_mean']['sIoU']['mean']>0 for c in COHORTS)
        decision=dict(gate_passed=passed,next='audit_negative_miner_FNR' if passed else 'stop_before_negative_miner',
                      no_automatic_method_promotion=True)
    write(target/'ALL_SOURCE_RESULTS.json',rows);write(target/'SUMMARY.json',summary)
    write(target/'DECISION.json',decision)
    print('SCORED',phase,decision,flush=True)
    return decision


def run(limit=0):
    import numpy as np,torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_actionness_spatial_attribution_v1 import resume
    import scripts.run_actionness_spatial_attribution_v1 as old
    from methods.decota_final_simplified_v1.tensors import state_hash
    p=verify();torch.set_num_threads(4);torch.manual_seed(20260918);np.random.seed(20260918)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    snapshot=None;guard=None;model=None;current=None;mhash=None;start=time.time();done=0
    try:
        snapshot=pause();guard=lease()
        for phase in ('actuator','negative'):
            if phase=='negative' and not analyze('actuator',p)['gate_passed']:break
            ordinal={c:0 for c in COHORTS}
            for row in p['rows']:
                ordinal[row['cohort']]+=1
                f=OUT/phase/(row['key'].replace(':','_')+'.pt')
                if exists(f):continue
                if limit and done>=limit:return
                assert time.time()-start<p['max_gpu_seconds'],'Bounded GPU budget exceeded'
                if row['cohort']!=current:
                    if model is not None:
                        assert state_hash(model.state_dict())==mhash
                        del model;gc.collect();torch.cuda.empty_cache()
                    model=model_load(row['cohort']);current=row['cohort'];mhash=state_hash(model.state_dict())
                versions={n:v._version for n,v in model.state_dict(keep_vars=True).items()}
                status(OUT/'STATUS.json',dict(status='running',phase=phase,key=row['key'],
                      ordinal=ordinal,run_done=done,pid=os.getpid(),updated=time.time()))
                torch.cuda.reset_peak_memory_stats()
                z=episode(model,row,p,phase,full_check=ordinal[row['cohort']]<=2)
                assert versions=={n:v._version for n,v in model.state_dict(keep_vars=True).items()}
                commit(f,z);done+=1;print('DONE',phase,row['key'],round(z['seconds'],2),'seconds',flush=True)
                del z;gc.collect();torch.cuda.empty_cache()
            d=analyze(phase,p)
            if model is not None:assert state_hash(model.state_dict())==mhash
        verify();status(OUT/'STATUS.json',dict(status='oracles_completed',decision=d,done=done,
                    pid=os.getpid(),seconds=time.time()-start,updated=time.time()))
    except BaseException as e:
        write(OUT/f'FAILURE_{int(time.time())}.json',dict(error=repr(e),time=time.time(),done=done))
        status(OUT/'STATUS.json',dict(status='failed',error=repr(e),done=done,updated=time.time()));raise
    finally:
        if model is not None:del model;gc.collect();torch.cuda.empty_cache()
        if guard is not None:guard.close()
        old.OUT=OUT;resume(snapshot)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','score'])
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--phase',default='actuator')
    a=parser.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run(a.limit)
    else:analyze(a.phase,verify())
