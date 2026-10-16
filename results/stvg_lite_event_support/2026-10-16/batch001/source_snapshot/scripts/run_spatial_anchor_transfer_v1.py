"""Replay fixed spatial parameters and perturb only expert reference geometry."""
import argparse,gc,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
OUT=ROOT/'artifacts/spatial_anchor_transfer_v1'
PARENT=ROOT/'artifacts/decota_boundary_anchor_p0_v1'
CODE=['scripts/run_spatial_anchor_transfer_v1.py','scripts/analyze_spatial_anchor_transfer_v1.py',
      'vg_tta/spatial_anchor_transfer_v1.py','protocols/spatial_anchor_transfer_v1.md']


def prepare():
    if (OUT/'LOCK.json').exists():return verify()
    p=read(PARENT/'LOCK.json');rows=[]
    prod=read(ROOT/'artifacts/reconstruction_hypothesis_ranking_v1/LOCK.json')['production_pins']
    for r in p['rows']:
        f=PARENT/'spatial'/f'{r["key"].replace(":","_")}.pt'
        c=ROOT/'artifacts/spatial_support_source_p0_v1/target_cache/correct'/f.name
        assert sha(f)==read(f.with_suffix('.json'))['sha256'];assert sha(r['path'])==r['sha256']
        assert sha(c)==read(c.with_suffix('.json'))['sha256']
        rows.append(dict(r,single=str(f),single_sha256=sha(f),cache=str(c),cache_sha256=sha(c)))
    donors={}
    for cohort in p['configurations']:
        rr=sorted([r for r in rows if r['cohort']==cohort],key=lambda r:(r['source'],r['key']))
        for i,r in enumerate(rr):donors[r['key']]=rr[(i+1)%len(rr)]['key']
    write(OUT/'LOCK.json',dict(rows=rows,donors=donors,configs=p['configurations'],labels=p['labels'],labels_sha256=p['labels_sha256'],
        code={f:sha(ROOT/f) for f in CODE},production_pins=prod,created=time.time(),max_seconds=3600,GT_online=False))
    return verify()


def verify():
    p=read(OUT/'LOCK.json')
    pins={**p['code'],**p['production_pins']}
    revision=OUT/'CODE_REVISION_1.json'
    if revision.exists():
        rv=read(revision)
        for f,h in rv['old'].items():assert pins[f]==h
        pins.update(rv['new'])
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    return p


def run():
    import torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import config
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from methods.decota_final_simplified_v1.optim import _fit
    from methods.decota_final_simplified_v1.objectives import SpatialLoss
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.backbone import full_prediction,make_batch,query_subject
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_anchor_transfer_v1 import perturb,transfer_state,PERTURBATIONS
    import scripts.run_spatial_query_negative_oracle_v1 as pauser
    import scripts.run_actionness_spatial_attribution_v1 as resumer
    p=verify();selected={r['key']:r for r in read(OUT/'INTERVENTION_LOCK.json')['rows']}
    lookup={r['key']:r for r in p['rows']};states={};initials={}
    for r in p['rows']:
        assert sha(r['path'])==r['sha256'];x=load(r['path']);states[r['key']]=x['spatial']['state'];initials[r['key']]=x['spatial']['initial_state']
    guard=None;snapshot=None;model=None;begin=time.time();receipts={};fits=0;backwards=0
    torch.set_num_threads(4);torch.manual_seed(20260919)
    try:
        pauser.OUT=OUT;snapshot=pauser.pause();guard=lease()
        for cohort in p['configs']:
            model=model_load(cohort);mh=state_hash(model.state_dict());first=True;lr0done=False
            rr=[r for r in p['rows'] if r['cohort']==cohort]
            for r in rr:
                dest=OUT/'runs'/f'{r["key"].replace(":","_")}.pt'
                if dest.with_suffix('.json').exists():
                    assert sha(dest)==read(dest.with_suffix('.json'))['sha256'];receipts[str(dest)]=sha(dest);continue
                assert time.time()-begin<p['max_seconds'];assert sha(r['cache'])==r['cache_sha256']
                c=load(r['cache']);x=load(r['path']);old=load(r['single']);ids=c['frame_ids']
                s=SpatialReplay(model,detached(c['views'],'cuda'),len(ids));initial=detached(s.initial)
                assert torch.equal(s.zero['boxes'].cpu(),x['native_boxes'])
                assert all(torch.equal(v.cpu(),initials[r['key']][k]) for k,v in initial.items())
                own=states[r['key']];s.restore(own)
                with torch.no_grad():v=s.values()
                assert torch.equal(v['boxes'].cpu(),x['predictions']['Full_DeCoTA']['boxes'])
                outputs={'Native':dict(boxes=s.zero['boxes'].cpu()),'Own':dict(boxes=v['boxes'].cpu(),state=own)}
                donor=p['donors'][r['key']];assert lookup[donor]['source']!=r['source']
                for name,matched in [('Donor',False),('Donor_normmatched',True)]:
                    state,meta=transfer_state(initial,states[donor],initials[donor],own,matched);s.restore(state)
                    with torch.no_grad():vv=s.values()
                    outputs[name]=dict(boxes=vv['boxes'].cpu(),state=detached(state,'cpu'),donor=donor,**meta)
                other=[q['key'] for q in rr if q['source']!=r['source']]
                mean={k:initial[k]+torch.stack([states[q][k].to(initial[k])-initials[q][k].to(initial[k]) for q in other]).mean(0) for k in initial}
                state,meta=transfer_state(initial,mean,initial,own,True);s.restore(state)
                with torch.no_grad():vv=s.values()
                outputs['Mean_LOSO_normmatched']=dict(boxes=vv['boxes'].cpu(),state=detached(state,'cpu'),donors=other,**meta)
                interventions={};audit=dict(native_exact=True,own_exact=True,temporal_fixed=True,reset_exact=False,full_reinsertion=False)
                if r['key'] in selected:
                    ai=selected[r['key']]['anchor_index'];a=old['fits'][ai]['anchor'];cfg=config(cohort)
                    for name in ['repeat']+PERTURBATIONS:
                        aa,info=(a,{}) if name=='repeat' else perturb(a,name)
                        loss=SpatialLoss([aa],s.zero['boxes'])
                        fit=_fit(s,s.zero,lambda v:4*loss(v['boxes']),lr=cfg.spatial_lr,steps=10,temporal=False,trace=True)
                        assert fit['failure'] is None;fits+=1;backwards+=fit['backwards']
                        if name=='repeat':
                            assert torch.equal(fit['final']['boxes'].cpu(),old['fits'][ai]['prediction']['boxes'])
                            assert fit['selected_step']==old['fits'][ai]['selected_step']
                            assert all(torch.equal(v.cpu(),old['fits'][ai]['state'][k]) for k,v in fit['state'].items())
                        interventions[name]=dict(anchor=aa,info=info,boxes=fit['final']['boxes'].cpu(),state=detached(fit['state'],'cpu'),
                            path=fit['path'],losses=fit['losses'],selected_step=fit['selected_step'],backwards=fit['backwards'])
                    if not lr0done:
                        loss=SpatialLoss([a],s.zero['boxes']);z=_fit(s,s.zero,lambda v:4*loss(v['boxes']),lr=0,steps=1,temporal=False)
                        assert torch.equal(z['final']['boxes'],s.zero['boxes']);audit['lr0']=True;backwards+=z['backwards'];lr0done=True
                if first:
                    from vg_tta.spatial_three_rounds_v1 import temporal_state
                    frames,fullids=decode(x['input']);batch=make_batch(frames,fullids,x['input'],model)
                    ts=temporal_state(x,config(cohort).eta);test=outputs['Donor_normmatched']
                    expected=dict(boxes=test['boxes'],logits=x['predictions']['Full_DeCoTA']['logits'])
                    with query_subject(model,batch,x['parses']['subject']):
                        full=full_prediction(model,batch,fullids,c['records'],{**test['state'],**ts},expected)
                    assert full['indices']==x['predictions']['Full_DeCoTA']['indices'];audit['full_reinsertion']=True
                    del batch,frames;first=False
                s.restore(initial)
                with torch.no_grad():v=s.values()
                assert torch.equal(v['boxes'],s.zero['boxes']);audit['reset_exact']=True
                assert all(v.grad is None and not v.requires_grad for v in model.parameters())
                save(dest,dict(key=r['key'],cohort=cohort,source=r['source'],frame_ids=ids,indices=x['predictions']['Full_DeCoTA']['indices'],
                    physical_interval=x['predictions']['Full_DeCoTA']['physical_interval'],outputs=outputs,interventions=interventions,audit=audit,GT_online=False))
                write(dest.with_suffix('.json'),dict(sha256=sha(dest),lock_sha256=sha(OUT/'LOCK.json')));receipts[str(dest)]=sha(dest)
                status(OUT/'STATUS.json',dict(status='running',done=len(receipts),total=64,fits=fits,seconds=time.time()-begin))
                print('DONE',len(receipts),r['key'],'fits',fits,flush=True);del s,x,c,old,v,outputs,interventions
            assert state_hash(model.state_dict())==mh;del model;model=None;gc.collect();torch.cuda.empty_cache()
        verify();write(OUT/'PREDICTION_BARRIER.json',dict(files=receipts,GT_online=False,created=time.time()))
        status(OUT/'STATUS.json',dict(status='completed',done=len(receipts),fits=fits,backwards=backwards,seconds=time.time()-begin))
    except BaseException as e:
        status(OUT/'FAILURE.json',dict(error=str(e),type=type(e).__name__,time=time.time()));raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
        if guard:guard.close()
        resumer.OUT=OUT;resumer.resume(snapshot)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);x=a.parse_args();prepare() if x.command=='prepare' else run()
