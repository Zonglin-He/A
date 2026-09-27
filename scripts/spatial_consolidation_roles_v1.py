"""Finite A→B→C execution; budget phase requires reviewed version lock."""
import argparse, gc, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
OUT=ROOT/'artifacts/spatial_consolidation_roles_v1'
PARENT=ROOT/'artifacts/spatial_online_long_v1'
DUAL={'hcstvg1_test':ROOT/'artifacts/spatial_slow_fast_hc_v1',
      'vidstg_test':ROOT/'artifacts/spatial_slow_fast_closed_loop_v1'}
CASES=['vidstg_test:008022','vidstg_test:008700']


def prepare():
    if (OUT/'LOCK.json').exists():return verify()
    p=read(PARENT/'LOCK.json')
    own=['scripts/spatial_consolidation_roles_v1.py','vg_tta/spatial_consolidation_v1.py',
         'protocols/spatial_consolidation_roles_v1.md','vg_tta/spatial_online_state_v1.py',
         'scripts/run_spatial_ssl_gpu_v1.py','scripts/run_spatial_regression_alignment_v1.py']
    dual={c:dict(root=str(d),lock_sha=sha(d/'LOCK.json'),barrier_sha=sha(d/'BARRIER.json'),
          files=read(d/'BARRIER.json')['files']) for c,d in DUAL.items()}
    assert all(read(d/'STATUS.json')['status']=='completed_scored_audited' for d in DUAL.values())
    write(OUT/'LOCK.json',dict(rows=p['rows'],streams=p['streams'],dual=dual,
          parent_sha=sha(PARENT/'LOCK.json'),pins={**p['production_pins'],**{f:sha(ROOT/f) for f in own}},
          labels=p['labels'],labels_sha256=p['labels_sha256'],alphas=[0.,1/16,1.],
          lr={'hcstvg1_test':.005,'vidstg_test':.05},steps=10,cases=CASES,
          prior_prefix=8,max_seconds=14400,max_bytes=20*1024**3,GT_online=False,
          historical_exposure=True,budget_stage='review_and_version_lock_required',created=time.time()))
    return verify()


def verify():
    p=read(OUT/'LOCK.json');assert sha(PARENT/'LOCK.json')==p['parent_sha']
    pins=dict(p['pins'])
    if (OUT/'PRELAUNCH_CODE_AMENDMENT.json').exists():
        a=read(OUT/'PRELAUNCH_CODE_AMENDMENT.json');assert a['original_lock_sha']==sha(OUT/'LOCK.json')
        pins.update(a['pins'])
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    for c,d in p['dual'].items():
        assert sha(Path(d['root'])/'LOCK.json')==d['lock_sha']
        assert sha(Path(d['root'])/'BARRIER.json')==d['barrier_sha']
    return p


def old_dual(p,c,seed,pos):
    f=Path(p['dual'][c]['root'])/'streams'/str(seed)/f'{pos:04d}.pt'
    assert sha(f)==p['dual'][c]['files'][str(f)]
    return load(f)


def setup():
    import torch
    torch.set_num_threads(4);torch.manual_seed(20260920)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def equivalence():
    if (OUT/'equivalence/BARRIER.json').exists():return
    import torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_consolidation_v1 import consolidate,QUERY
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    p=verify();setup();guard=lease();files={};stats=[];model=None
    try:
        for c in DUAL:
            model=model_load(c);mh=state_hash(model.state_dict())
            for seed,keys in p['streams'][c].items():
                positions=[i for i,k in enumerate(keys) if k in CASES] if c=='vidstg_test' else [0]
                for pos in positions:
                    f=OUT/'equivalence'/c/seed/f'{pos:04d}.pt'
                    if f.with_suffix('.json').exists():
                        rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'];z=load(f)
                        stats.append(z['stats']);files[str(f)]=sha(f);continue
                    old=old_dual(p,c,seed,pos);initial=old['initial'];cons=consolidate(initial,old['fast']['state'],1/16)
                    names=[n for n in initial if n!=QUERY]
                    a=torch.cat([(old['slow']['state'][n]-initial[n]).flatten().double() for n in names])
                    b=torch.cat([(cons[n]-initial[n]).flatten().double() for n in names])
                    cosine=float(a@b/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None
                    ratio=float(b.norm()/a.norm()) if a.norm()>0 else None
                    nextkey=keys[pos+1];row=p['rows'][nextkey];assert sha(row['path'])==row['sha256'];x=load(row['path'])
                    frames,ids=decode(x['input']);assert ids==x['frame_ids'];batch,records,r=frozen_forward(model,frames,x)
                    source=detached(r.initial);assert torch.equal(r.zero['boxes'].cpu(),x['native_boxes']);preds={}
                    for tag,st in [('Slow',old['committed']),('Consolidated',cons)]:
                        r.restore(st)
                        with torch.no_grad():v=detached(r.values())
                        with query_subject(model,batch,x['parses']['subject']):full_prediction(model,batch,ids,records,st,v)
                        preds[tag]=detached(v,'cpu')
                    nxt=old_dual(p,c,seed,pos+1)
                    assert torch.equal(preds['Slow']['boxes'],nxt['before'])
                    delta=(preds['Slow']['boxes']-preds['Consolidated']['boxes']).abs()
                    st=dict(cohort=c,seed=int(seed),position=pos,key=keys[pos],nextkey=nextkey,
                        LN_cosine=cosine,LN_norm_ratio=ratio,slow_norm=float(a.norm()),cons_norm=float(b.norm()),
                        before_box_mean_abs=float(delta.mean()),before_box_max_abs=float(delta.max()),
                        boxes_equal=torch.equal(preds['Slow']['boxes'],preds['Consolidated']['boxes']),GT_online=False)
                    r.restore(source);assert state_hash(model.state_dict())==mh
                    save(f,dict(stats=st,predictions=preds,slow=old['committed'],consolidated=cons))
                    write(f.with_suffix('.json'),dict(sha256=sha(f)));files[str(f)]=sha(f);stats.append(st)
                    print('equivalence',c,seed,pos,st,flush=True)
                    del r,batch,frames,x,old,nxt;gc.collect()
            del model;model=None;gc.collect();torch.cuda.empty_cache()
        assert len(files)==9
        write(OUT/'equivalence/SUMMARY.json',dict(rows=stats,full_reinsertions=18,GT_online=False))
        write(OUT/'equivalence/BARRIER.json',dict(files=files,GT_online=False))
    finally:
        del model;gc.collect();torch.cuda.empty_cache();guard.close()


def run(stage,smoke=False):
    import torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward,config
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_online_state_v1 import arrival,QUERY
    from vg_tta.spatial_consolidation_v1 import consolidate,scoped_fit
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    p=verify();assert (OUT/'equivalence/BARRIER.json').exists()
    if stage=='roles' and not smoke:assert (OUT/'single/SUMMARY.json').exists()
    setup();guard=lease();files={};model=None;start=time.time();total_bytes=0
    root=OUT/stage;arms=['C1','C2'] if stage=='single' else ['Query','LN']
    try:
        for c in DUAL:
            model=model_load(c);mh=state_hash(model.state_dict());cfg=config(c)
            assert cfg.spatial_lr==p['lr'][c] and cfg.spatial_steps==10
            streams=list(p['streams'][c].items())[:1] if smoke else p['streams'][c].items()
            for seed,keys in streams:
                previous={a:None for a in arms};prevhash=None
                for pos,key in enumerate(keys[:2] if smoke else keys):
                    assert time.time()-start<p['max_seconds']
                    f=root/'streams'/c/seed/f'{pos:04d}.pt'
                    if f.with_suffix('.json').exists():
                        rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'] and rc['lock']==sha(OUT/'LOCK.json')
                        z=load(f);assert z['previous_file_sha256']==prevhash and z['key']==key
                        if stage=='single':previous={a:z['fits'][a]['committed'] for a in arms}
                        prevhash=sha(f);files[str(f)]=prevhash;total_bytes+=f.stat().st_size;continue
                    row=p['rows'][key];assert sha(row['path'])==row['sha256'];x=load(row['path'])
                    old=old_dual(p,c,seed,pos)
                    frames,ids=decode(x['input']);assert ids==x['frame_ids'];batch,records,r=frozen_forward(model,frames,x)
                    source=detached(r.initial);assert torch.equal(r.zero['boxes'].cpu(),x['native_boxes'])
                    assert state_hash(source)==state_hash(old['source_state'])
                    fits={};audits=Counter();is_case=key in CASES;doaudit=pos in [0,1,8,len(keys)//2,len(keys)-1] or is_case
                    for arm in arms:
                        initial=arrival(source,previous[arm],'O-split') if stage=='single' else old['initial']
                        r.initial=detached(initial,'cuda');r.restore(r.initial)
                        with torch.no_grad():before=detached(r.values())
                        if stage=='roles':assert torch.equal(before['boxes'].cpu(),old['before'])
                        scope='full' if stage=='single' else ('query' if arm=='Query' else 'ln')
                        torch.cuda.synchronize();t=time.perf_counter()
                        fit=scoped_fit(r,before,x['spatial']['anchors'],cfg,scope,trace=is_case)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-t
                        assert fit['failure'] is None,fit['failure']
                        if not x['spatial']['anchors']:assert state_hash(fit['state'])==state_hash(initial)
                        if doaudit:
                            with query_subject(model,batch,x['parses']['subject']):full_prediction(model,batch,ids,records,fit['state'],fit['final'])
                            audits['full_reinsertions']+=1
                        fit=detached(fit,'cpu');fit.update(before=before['boxes'].cpu(),seconds=elapsed)
                        if stage=='single':
                            alpha=1/16 if arm=='C1' else 1.
                            fit['committed']=consolidate(fit['initial_state'],fit['state'],alpha)
                            previous[arm]=fit['committed']
                            assert torch.count_nonzero(fit['committed'][QUERY])==0
                            if pos==0:
                                assert torch.equal(fit['final']['boxes'],x['predictions']['Full_DeCoTA']['boxes'])
                                audits['episodic_exact']+=1
                        else:
                            frozen=[n for n in source if (n!=QUERY if arm=='Query' else n==QUERY)]
                            assert all(torch.equal(fit['state'][n],fit['initial_state'][n]) for n in frozen)
                            assert fit['trainable_parameters']==(256 if arm=='Query' else 1536)
                            audits['scope_frozen_exact']+=1
                        fits[arm]=fit
                    if pos==0:
                        r.initial=detached(source,'cuda');r.restore(r.initial)
                        z0=scoped_fit(r,r.zero,x['spatial']['anchors'],replace(cfg,spatial_lr=0,spatial_steps=1))
                        assert state_hash(z0['state'])==state_hash(source) and torch.equal(z0['final']['boxes'].cpu(),x['native_boxes'])
                        audits['lr0']+=1
                    r.restore(source)
                    z=dict(key=key,source=row['source'],cohort=c,seed=int(seed),position=pos,
                        source_state=detached(source,'cpu'),fits=fits,previous_file_sha256=prevhash,
                        native=x['native_boxes'],episodic=x['predictions']['Full_DeCoTA']['boxes'],
                        dual=old['fast']['final']['boxes'],dual_before=old['before'],
                        frame_ids=ids,indices=x['predictions']['Full_DeCoTA']['indices'],
                        observed=x['actual_observation_positions'],anchors=x['spatial']['anchors'],
                        audit=dict(audits),GT_online=False)
                    save(f,z);prevhash=sha(f);write(f.with_suffix('.json'),dict(sha256=prevhash,lock=sha(OUT/'LOCK.json')))
                    files[str(f)]=prevhash;total_bytes+=f.stat().st_size;assert total_bytes<p['max_bytes']
                    status(root/'STATUS.json',dict(status='running',stage=stage,done=len(files),total=2370,
                        cohort=c,seed=seed,position=pos,seconds=time.time()-start,pid=os.getpid()))
                    print(stage,c,seed,pos,key,round(time.time()-start,2),flush=True)
                    del r,batch,frames,x,old,z,fits;gc.collect()
                assert state_hash(model.state_dict())==mh
            del model;model=None;gc.collect();torch.cuda.empty_cache()
        verify();write(root/('SMOKE.json' if smoke else 'BARRIER.json'),dict(files=files,GT_online=False,seconds=time.time()-start))
        status(root/'STATUS.json',dict(status='smoke_passed' if smoke else 'inference_complete',done=len(files)))
    except BaseException as e:
        write(root/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),done=len(files)))
        status(root/'STATUS.json',dict(status='failed',error=repr(e),done=len(files)));raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache();guard.close()


def score_stage(stage):
    import numpy as np
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.spatial_consolidation_v1 import consolidate,QUERY
    from scripts.analyze_spatial_reference_absorption_v1 import score
    from scripts.analyze_spatial_online_state_v1 import stat
    p=verify();root=OUT/stage;b=read(root/'BARRIER.json');assert len(b['files'])==2370
    assert sha(p['labels'])==p['labels_sha256'];labels=read(p['labels']);rows=[];cases=[];audit=Counter();hist={};prevhash={}
    for f,h in sorted(b['files'].items()):
        assert sha(f)==h;z=load(f);c=z['cohort'];seed=z['seed'];sid=(c,seed)
        assert z['previous_file_sha256']==prevhash.get(sid);prevhash[sid]=h
        if stage=='roles':old=old_dual(p,c,seed,z['position'])
        boxes=dict(Native=z['native'],Episodic=z['episodic'],Dual=z['dual'],DualBefore=z['dual_before'])
        for arm,fit in z['fits'].items():
            initial=fit['initial_state'];expected=(z['source_state'] if z['position']==0 else hist[(c,seed,arm)]) if stage=='single' else old['initial']
            assert state_hash(initial)==state_hash(expected)
            if stage=='single':
                exp=consolidate(initial,fit['state'],1/16 if arm=='C1' else 1.)
                assert state_hash(exp)==state_hash(fit['committed']);hist[(c,seed,arm)]=exp
            else:
                assert all(__import__('torch').equal(initial[n],fit['state'][n]) for n in initial if (n!=QUERY if arm=='Query' else n==QUERY))
            boxes[arm]=fit['final']['boxes'];boxes[arm+'Before']=fit['before'];audit['states_verified']+=1
            if z['key'] in CASES:
                for v in fit['path']:
                    cases.append(dict(key=z['key'],seed=seed,position=z['position'],arm=arm,step=v['step'],loss=v['loss'],
                        query_norm=float(v['state'][QUERY].norm()),
                        ln_displacement=sum(float((v['state'][n]-initial[n]).double().square().sum()) for n in initial if n!=QUERY)**.5,
                        metrics=score(v['boxes'],labels[z['key']],z['frame_ids'],z['indices'],z['observed'],z['anchors'])))
        metrics={a:score(v,labels[z['key']],z['frame_ids'],z['indices'],z['observed'],z['anchors']) for a,v in boxes.items()}
        assert len({v['tIoU'] for v in metrics.values()})==1
        audit.update(z['audit'])
        rows.append({k:z[k] for k in ['key','source','cohort','seed','position']}|dict(metrics=metrics,anchors=len(z['anchors']),
             fits={a:{k:v[k] for k in ['selected_step','backwards','seconds','scope','trainable_parameters']} for a,v in z['fits'].items()}))
    summaries={};ms=['sIoU','vIoU_corrected','tIoU','unobserved_sIoU']
    pairs=([('C1','Episodic'),('C1','Dual'),('C2','Episodic'),('C1','C2'),('C1Before','Native'),('C2Before','Native'),('C1','Native'),('C2','Native'),('Episodic','Native')]
           if stage=='single' else [('Query','Dual'),('LN','Dual'),('Query','Episodic'),('LN','Episodic'),('Query','Native'),('LN','Native'),('Dual','Native')])
    def comp(a,bb,rr):
        g=defaultdict(list)
        for r in rr:g[r['source']].append(r)
        return {m:stat([np.mean([r['metrics'][a][m]-r['metrics'][bb][m] for r in gg]) for gg in g.values()]) for m in ms}
    for c in DUAL:
        rr=[r for r in rows if r['cohort']==c and r['position']>=8];sources={r['source'] for r in rr}
        assert len(sources)==(50 if c=='hcstvg1_test' else 724) and len(rr)==len(sources)*3
        assert set(Counter(r['source'] for r in rr).values())=={3}
        ss=dict(sources=len(sources),arrivals=len(rr),absolute={a:{m:float(np.mean([r['metrics'][a][m] for r in rr])) for m in ms} for a in rr[0]['metrics']},
             contrasts={a+'-'+bb:comp(a,bb,rr) for a,bb in pairs},
             orders={str(s):{a+'-'+bb:comp(a,bb,[r for r in rr if r['seed']==s]) for a,bb in pairs} for s in sorted({r['seed'] for r in rr})})
        if stage=='roles':
            ss['good_before_descriptive']={}
            for m in ['sIoU','vIoU_corrected']:
                grouped={s:[r for r in rr if r['source']==s] for s in sources}
                selected=[g for g in grouped.values() if np.mean([r['metrics']['DualBefore'][m]-r['metrics']['Native'][m] for r in g])>.001]
                ss['good_before_descriptive'][m]=dict(n=len(selected),arms={a:dict(
                    below_native_count=int(sum(np.mean([r['metrics'][a][m]-r['metrics']['Native'][m] for r in g])<-.001 for g in selected)),
                    loss_vs_before_count=int(sum(np.mean([r['metrics'][a][m]-r['metrics']['DualBefore'][m] for r in g])<-.001 for g in selected))) for a in ['Dual','Query','LN']})
        summaries[c]=ss
    write(root/'ALL_SOURCE_RESULTS.json',rows);write(root/'SUMMARY.json',summaries);write(root/'CASES.json',cases)
    write(root/'AUDIT.json',dict(status='passed',counts=dict(audit),GT_online=False,production_unchanged=True))
    report=[f'# {stage} results','', 'Fixed formal time; historical exposure;50HC/724Vid sources,3histories. Native is NOT full Frozen. No automatic promotion.']
    for c,ss in summaries.items():
        report+=['',f'## {c}','', '|Arm|sIoU|tIoU|vIoU|','|---|---:|---:|---:|']
        for a,v in ss['absolute'].items():report.append(f"|{a}|{v['sIoU']*100:.3f}|{v['tIoU']*100:.3f}|{v['vIoU_corrected']*100:.3f}|")
        for pair,d in ss['contrasts'].items():
            v=d['vIoU_corrected'];report.append(f"{pair}: Δv {v['mean']*100:+.3f}pp, conditionalCI {np.array(v['ci95_conditional'])*100}.")
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    status(root/'STATUS.json',dict(status='completed_scored_audited',done=2370))
    print('SCORED',stage,{c:s['contrasts'][pairs[0][0]+'-'+pairs[0][1]] for c,s in summaries.items()},flush=True)


def execute():
    started=time.time();status(OUT/'STATUS.json',dict(status='running',stage='equivalence',pid=os.getpid()))
    equivalence()
    for stage in ['single','roles']:
        assert time.time()-started<43200
        status(OUT/'STATUS.json',dict(status='running',stage=stage,pid=os.getpid()))
        if not (OUT/stage/'BARRIER.json').exists():run(stage)
        if not (OUT/stage/'SUMMARY.json').exists():score_stage(stage)
    status(OUT/'STATUS.json',dict(status='stages_A_B_C_complete_budget_review_required',seconds=time.time()-started,
           budget_stage='not_started',production_unchanged=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','equivalence','smoke','execute','score']);ap.add_argument('--stage',choices=['single','roles'],default='single');a=ap.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='equivalence':equivalence()
        elif a.mode=='smoke':equivalence();run('single',True);run('roles',True)
        elif a.mode=='score':score_stage(a.stage)
        else:execute()
    except BaseException as e:
        status(OUT/'STATUS.json',dict(status='failed',error=repr(e),stage=a.mode));raise
