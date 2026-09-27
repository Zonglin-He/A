"""Finite online-corruption experiment runner. Each condition seals before GT."""
import argparse,copy,fcntl,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import *
from methods.decota_v1 import decode,fit
from vg_tta.metrics import interval_from_logits
from vg_tta.native_coverage_calibration_v1 import native_at_length
from vg_tta.posterior_mass_coverage_v1 import select as mass_select

def prior_predictions(z,ids,native,boxes):
    pm=mass_select(z,ids,tau=.9)['indices'];fixed=native_at_length(z,ids,fraction=.75)['indices']
    return {'posterior_mass90':{'boxes':boxes,'indices':tuple(pm)},'fixed75':{'boxes':boxes,'indices':tuple(fixed)}}

def lift(result,positions,ids):
    from vg_tta.tastvg_corruption_alignment_v1 import align_native_prediction
    for name,r in result['predictions'].items():
        x=align_native_prediction({'pred_boxes':r['boxes'],'pred_sted':result['native_logits'].reshape(len(positions),2),'pred_indices':r['indices']},positions,ids)
        r.update(boxes=x['pred_boxes'],indices=tuple(x['pred_indices']))
    result['native_frame_ids']=result['frame_ids'];result['frame_ids']=list(ids);result['retained_positions']=positions
    return result

def tube_episode(model,q,c,spec,snapshot,*,ablate,parent):
    from vg_tta.unanchored_shift_predictor_v1 import predict_episode
    from vg_tta.dense_expansion_data_v1 import decode_raw
    previous=Path(parent)/spec['group']/c['name']/'predictions'/f'{int(q["index"]):06d}.pt'
    if previous.exists() and previous.with_suffix('.json').exists():
        receipt=read(previous.with_suffix('.json'));assert sha(previous)==receipt['bundle_sha256'];b=load(previous)
        provenance={'reused_exact_parent':str(previous),'sha256':receipt['bundle_sha256']}
    else:
        raw,ids=decode_raw(q)
        b=predict_episode(model,raw,ids,q['caption'],f'{spec["group"]}:{q["index"]}',corruption=c['corruption'],severity=c['severity'],seed=c['seed'],
            selected_config=spec['config'],old_config=None,matched_anchor_gamma=1e-4 if ablate else None,
            check_controls=False,source_snapshot=snapshot,device='cuda:0',baselines=True,sar_entropy_margin_fraction=.4)
        provenance={'new_online_corruption':True};del raw
    pos=b['frame_mapping']['retained_positions'];ids=b['frame_mapping']['retained_frame_ids'];n=b['native_predictions'];z=n['frozen']['pred_sted'];boxes=n['frozen']['pred_boxes']
    native=interval_from_logits(z);extent=interval_from_logits(n['ours']['pred_sted']);pred={}
    for name,m in [('frozen','frozen'),('coupled_coverage','ours'),('tent','tent'),('memo','memo'),('sar','sar')]:
        pred[name]={'boxes':n[m]['pred_boxes'],'indices':interval_from_logits(n[m]['pred_sted'])}
    pred['decota']={'boxes':boxes,'indices':decode(z,extent,ids,native_indices=native)['indices']}
    pred.update(prior_predictions(z,ids,native,boxes));audits={'original':b['audits']};times=b['timing']
    h=b['head_input'].cuda();cfg=spec['config']
    if ablate:
        configs={'anchor1e4':{**cfg,'gamma':1e-4},'lr0':{**cfg,'lr':0.},**{f'steps{s}':{**cfg,'steps':s} for s in [0,1,3,10,20]}}
        for name,cc in configs.items():
            t=time.perf_counter();head,zs,audit=fit(model.sted_embed,[h],backbone='tubedetr',lr=cc['lr'],steps=cc['steps'],gamma=cc.get('gamma',0.))
            ij=interval_from_logits(zs[0]);pred[name]={'boxes':boxes,'indices':decode(z,ij,ids,native_indices=native)['indices']};audits[name]=audit;times[name+'_seconds']=time.perf_counter()-t
            if name in ['lr0','steps0']:assert pred[name]['indices']==native
            del head,zs
    result={'predictions':pred,'native_logits':z.detach().cpu(),'frame_ids':ids,'head_inputs':[h.detach().cpu()],
        'fitted_head_state':b['fitted_head_states']['ours'],'audits':audits,'timing':times,'GT_used':False,'provenance':provenance,
        'corruption':b['corruption'],'realized_corruption':b['realized_corruption']}
    return lift(result,pos,q['frame_ids'])

def run(backbone,cohort,condition,limit=0):
    from vg_tta.dense_expansion_data_v1 import decode_raw
    from vg_tta.foreground_runtime import state_digest
    p=plan();locksha=sha(OUT/'lock.json');dev=cohort.startswith('dev_')
    if dev:
        g=cohort[4:];qs=p['dev'][g];spec={'group':g,'config':{'lr':.001,'steps':5,'gamma':0.}}
    else:spec=p['cohorts'][cohort];g=spec['group'];qs=spec['queries']
    c=next(c for c in p['conditions'] if c['name']==condition);dest=folder(backbone,cohort,condition)
    if (dest/'barrier.json').exists():return
    if shutil.disk_usage(ROOT).free<80*2**30:raise RuntimeError('Safe disk threshold: fewer than80GiB free')
    device='cuda:0';torch.set_num_threads(4);torch.manual_seed(20260909);np.random.seed(20260909)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX)
    if backbone=='tastvg':
        from scripts import run_tastvg_span_tta as ta
        from vg_tta.decota_tastvg_episode_v1 import make_batch,episode
        from vg_tta.shift_corruptions_v2 import apply_corruption
        source='hcstvg2' if g=='hc_to_vid' else 'vidstg';ckpt=p['ta_checkpoints'][source];assert sha(ckpt['path'])==ckpt['sha256']
        model,_,report=ta.load_model_on_device(source,ROOT/'artifacts/tastvg_runtime'/('hc-stvg2' if source=='hcstvg2' else 'vidstg'),checkpoint=Path(ckpt['path']),device=device,source_dataset=source)
        model.eval().requires_grad_(False);before=state_digest(model)
        cfg=spec['config'] if dev else read(p['tastvg_selection'])[g]['config']
        hp=OUT/f'heads/tastvg_{g}.pt'
        if not hp.exists():save(hp,copy.deepcopy(model.temp_embed).cpu())
    else:
        from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _lazy_runtime
        from scripts import evaluate_fullspan_scale_shift_external_v1 as external
        rt=_lazy_runtime();rt.add_repo_to_path(ROOT/'external/TubeDETR');model,_=rt.build_model(ROOT/'external/TubeDETR',device=device,resolution=224,stride=2)
        report=rt.load_official_checkpoint(model,spec['checkpoint']);assert not report['missing_keys'] and not report['unexpected_keys']
        model.eval().requires_grad_(False);snapshot=external._snapshot_full(model);before=state_digest(model)
        hp=OUT/f'heads/tubedetr_{g}.pt'
        if not hp.exists():save(hp,copy.deepcopy(model.sted_embed).cpu())
    ablate=dev or c['name']=='clean' or (c['severity']==3 and c['seed']==20260908)
    receipts=[];completed=0
    for k,r in enumerate(qs):
        q=r['input'];f=dest/'predictions'/f'{int(q["index"]):06d}.pt';rp=f.with_suffix('.json')
        if rp.exists():
            rr=read(rp);assert rr['lock_sha256']==locksha and sha(f)==rr['sha256'];receipts.append(rr);continue
        if f.exists():raise RuntimeError('Preserved orphan prediction: '+str(f))
        t=time.monotonic();torch.cuda.reset_peak_memory_stats()
        if backbone=='tubedetr':result=tube_episode(model,q,c,spec,snapshot,ablate=ablate,parent=spec['parent'])
        else:
            raw,ids=decode_raw(q)
            if c['corruption']=='clean':
                from vg_tta.unanchored_shift_predictor_v1 import _clean_corruption
                damaged=_clean_corruption(raw,ids,f'{g}:{q["index"]}',c['seed'])
            else:damaged=apply_corruption(raw,c['corruption'],c['severity'],query_id=f'{g}:{q["index"]}',seed=c['seed'],frame_ids=ids)
            pos=damaged.metadata['retained_positions'];kept=damaged.metadata['retained_frame_ids'];batch=make_batch(damaged.frames,kept,q,r['subject']['subject'],model)
            result=episode(model,batch,cfg,p['baseline_configs'],baselines=True,ablations=ablate)
            result['predictions'].update(prior_predictions(result['native_logits'],kept,result['predictions']['frozen']['indices'],result['predictions']['frozen']['boxes']))
            result['realized_corruption']=damaged.metadata;result=lift(result,pos,ids);del raw,damaged,batch
        result.update(group=g,cohort=cohort,index=q['index'],source=q['source'],condition=c,backbone=backbone,lock_sha256=locksha,
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,episode_seconds=time.monotonic()-t)
        save(f,result);rr={'index':q['index'],'source':q['source'],'path':str(f),'sha256':sha(f),'lock_sha256':locksha,'GT_used':False};write(rp,rr);receipts.append(rr)
        status(OUT/'cell_progress.json',{'backbone':backbone,'cohort':cohort,'condition':condition,'completed':k+1,'total':len(qs),'seconds':result['episode_seconds'],'time':time.time()})
        print('PREDICT',backbone,cohort,condition,k+1,len(qs),round(result['episode_seconds'],2),flush=True);del result;gc.collect();completed+=1
        if limit and completed>=limit:break
    assert state_digest(model)==before
    if len(receipts)==len(qs):write(dest/'barrier.json',{'status':'sealed','queries':len(qs),'receipts':receipts,'GT_used':False,'state_unchanged':True,'lock_sha256':locksha})

def donor(backbone,cohort):
    """Actual different-source fitted parameter transfer, no GT or new fitting."""
    p=plan();spec=p['cohorts'][cohort];g=spec['group'];dest=folder(backbone,cohort,'clean');bar=read(dest/'barrier.json')
    out=dest/'donor.pt'
    if out.exists():return
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX)
    if backbone=='tastvg':
        from scripts import run_tastvg_span_tta as ta
        from vg_tta.tastvg_baseline_expansion import replay_temporal_head_native as replay
        from vg_tta.decota_tastvg_episode_v1 import fitted_merge
        ta.official_imports()
    else:
        from vg_tta.tubedetr_runtime import add_repo_to_path
        add_repo_to_path(ROOT/'external/TubeDETR')
        from methods.decota_v1._fullspan import replay_temporal_head as replay
    torch.set_num_threads(4);head=load(OUT/f'heads/{backbone}_{g}.pt').cuda().eval();rs=bar['receipts'];rows={}
    for i,r in enumerate(rs):
        j=next((i+d)%len(rs) for d in range(1,len(rs)) if rs[(i+d)%len(rs)]['source']!=r['source'])
        q=load(r['path']);d=load(rs[j]['path']);head.load_state_dict(d['fitted_head_state']);ids=q['native_frame_ids']
        with torch.no_grad():zs=[replay(head,h.cuda(),'cuda:0') for h in q['head_inputs']]
        extent=fitted_merge(zs,q['native_views'],ids) if backbone=='tastvg' else interval_from_logits(zs[0])
        pos=q['retained_positions'];inv={v:k for k,v in enumerate(pos)};native=tuple(inv[k] for k in q['predictions']['frozen']['indices'])
        ij=decode(q['native_logits'],extent,ids,native_indices=native)['indices']
        rows[str(r['index'])]={'indices':tuple(pos[k] for k in ij),'donor_index':rs[j]['index'],'donor_source':rs[j]['source'],'GT_used':False}
    save(out,rows);write(dest/'donor_receipt.json',{'sha256':sha(out),'queries':len(rows),'GT_used':False})

def batch_run():
    p=plan();jobs=[]
    for g in p['dev']:jobs.append(['run','--backbone','tastvg','--cohort','dev_'+g,'--condition','clean'])
    jobs.append(['@tune'])
    for c in p['conditions']:
        for b in ['tubedetr','tastvg']:
            for cohort in p['cohorts']:
                jobs.append(['run','--backbone',b,'--cohort',cohort,'--condition',c['name']])
                if c['name']=='clean':jobs.append(['donor','--backbone',b,'--cohort',cohort])
            jobs.append(['@score',b,c['name']])
    jobs.append(['@report'])
    if not (OUT/'jobs.json').exists():write(OUT/'jobs.json',jobs)
    else:assert read(OUT/'jobs.json')==jobs
    for i,args in enumerate(jobs):
        job=OUT/f'jobs/{i:04d}';done=job/'complete.json'
        if done.exists():continue
        job.mkdir(parents=True,exist_ok=True)
        if args[0]=='@tune':cmd=[sys.executable,'scripts/tune_decota_tastvg_v1.py']
        elif args[0]=='@score':cmd=[sys.executable,'scripts/score_decota_matrix_v1.py',*args[1:]]
        elif args[0]=='@report':cmd=[sys.executable,'scripts/score_decota_matrix_v1.py','report']
        else:cmd=[sys.executable,__file__,*args]
        status(OUT/'progress.json',{'status':'running','job':i,'total_jobs':len(jobs),'command':cmd,'time':time.time()})
        attempt=len(list(job.glob('attempt_*.log')));log=job/f'attempt_{attempt:02d}.log'
        with log.open('x') as f:proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        receipt={'command':cmd,'returncode':proc.returncode,'log':str(log),'log_sha256':sha(log),'time':time.time()}
        if proc.returncode:
            write(job/f'failure_{attempt:02d}.json',receipt);status(OUT/'progress.json',{'status':'failed','job':i,**receipt});raise RuntimeError('Preserved failed stage: '+str(log))
        write(done,receipt)
    status(OUT/'progress.json',{'status':'complete','jobs':len(jobs),'time':time.time()})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','donor','batch']);ap.add_argument('--backbone',choices=['tubedetr','tastvg']);ap.add_argument('--cohort');ap.add_argument('--condition',default='clean');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.backbone,a.cohort,a.condition,a.limit)
    elif a.stage=='donor':donor(a.backbone,a.cohort)
    else:batch_run()
