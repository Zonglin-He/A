"""Frozen 447-query Oracle–Mixer Gap Audit, write-once resumable episodes."""
import argparse,gc,json,os,shutil,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_joint_learnability import D as OLD,ROSTER,OUT,load,setup,example,ADAPTER_SHA,B1,B1_SHA,equal
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,local_dependencies,allocation,tensor_sha
D=OUT/'oracle_mixer_gap_v1';E=OLD/'evaluation';PROTOCOL=ROOT/'protocols/desta3d_v3_oracle_mixer_gap_v1.md'
SEEDS=['seed20260928','seed20260929'];ARMS=['B1',*SEEDS,'oracle']


def prepare():
    import torch,subprocess
    assert not D.exists() and sha(B1)==B1_SHA
    tests=['tests/test_desta3d_v3_oracle_mixer_gap.py','tests/test_external_qualification_metrics.py','tests/test_desta3d_v3_oracle_mixer_gap_score.py']
    result=subprocess.run([sys.executable,'-B','-m','pytest','-q',*tests],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0 and '10 passed' in result.stdout,(result.stdout,result.stderr)
    check_pins(read(OLD/'LOCK.json')['pins']);assert read(OLD/'ROOT_COMPLETION_SUMMARY.json')['status']=='completed_independently_audited_not_qualified'
    assert read(E/'COMPLETE.json')['seal_sha']==sha(E/'PREDICTIONS_SEAL.json')
    oldseal=read(E/'PREDICTIONS_SEAL.json');check_pins({str(E/p):h for p,h in oldseal['files'].items()})
    rows=read(OLD/'VALIDATION_INPUTS.json');assert len(rows)==447 and len({r['source'] for r in rows})==31
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True);support=[];elements=vocab=0
    for i,row in enumerate(rows):
        label=json.loads(db.execute('select labels_json from examples where key=?',(row['key'],)).fetchone()[0]);p=load(E/'episodes'/f'{i:04}'/'B1.pt')
        event=bool(sum(label['event_active']));spatial=bool(sum(label['box_valid'][int(j)] for j in p['positions']))
        support.append(dict(index=i,event_available=event,spatial_available=spatial))
        t,h,w=p['preprocess']['grid'][0];elements+=t*(h//2)*(w//2)*2560;vocab+=len(p['positions'])*4*152775*4
    cfg=dict(stage='source_447_oracle_mixer_gap',queries=447,parents=31,seed=20260928,radius=read(OLD/'CONFIG.json')['radius'],
        phase_seconds=3600,minimum_free_bytes=8*2**30,maximum_new_bytes=110*2**30,cap=None,optimizer_steps=0,
        source_GT_oracle=True,target_read=False,estimated_raw_bytes=elements*4*5+vocab,expected_backwards=sum(x['event_available']+x['spatial_available'] for x in support),
        data_status='previous confirmation now exposed diagnosis; not future heldout',missing_rule='unit(0)=0, remaining branch same fixed radius, both missing no-op')
    assert shutil.disk_usage(ROOT).free>cfg['maximum_new_bytes']+cfg['minimum_free_bytes']
    write(D/'CONFIG.json',cfg);write(D/'INPUTS.json',rows);write(D/'SUPPORT_PREFLIGHT.json',support)
    write(D/'CPU_PREFLIGHT.json',dict(status='passed',tests=10,GPU=False,output=result.stdout));shutil.copyfile(OLD/'BASIS.pt',D/'BASIS.pt')
    paths=local_dependencies([Path(__file__),PROTOCOL,D/'CONFIG.json',D/'INPUTS.json',D/'SUPPORT_PREFLIGHT.json',D/'CPU_PREFLIGHT.json',D/'BASIS.pt',
        ROOT/'vg_tta/desta3d_v3_oracle_mixer_gap.py',ROOT/'tests/test_desta3d_v3_oracle_mixer_gap.py',
        OLD/'LOCK.json',E/'LOCK.json',E/'PREDICTIONS_SEAL.json',E/'COMPLETE.json',E/'independent_readback_v1/REPORT.json',
        OLD/'VALIDATION_PROVENANCE.json',ROSTER/'SOURCE.sqlite',B1,*[OLD/a/'FINAL.pt' for a in SEEDS],
        ROOT/'scripts/score_desta3d_v3_oracle_mixer_gap.py',ROOT/'scripts/audit_desta3d_v3_oracle_mixer_gap_raw.py',
        ROOT/'scripts/supervise_desta3d_v3_oracle_mixer_gap.py',ROOT/'scripts/crosscheck_desta3d_v3_oracle_mixer_gap.py',
        ROOT/'tests/test_external_qualification_metrics.py',ROOT/'tests/test_desta3d_v3_oracle_mixer_gap_score.py'])
    write(D/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}));write(D/'REGISTRATION.json',dict(time=time.time(),protocol_sha=sha(PROTOCOL),status='registered_before_GPU',prior_seconds=63593.56801247615))
    print('GAP_REGISTERED',cfg,flush=True)


def run(name,limit):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    cfg=read(D/'CONFIG.json');d=OUT/name;assert not d.exists()
    check_pins(read(D/'LOCK.json')['pins']);write(d/'CONFIG.json',{**cfg,'phase_seconds':900 if limit==1 else 3600})
    write(d/'LOCK.json',dict(pins={**read(D/'LOCK.json')['pins'],str(D/'LOCK.json'):sha(D/'LOCK.json')}))
    write(d/'REGISTRATION.json',dict(time=time.time(),query_limit=limit,optimizer_steps=0))
    with allocation(d) as (cfg,guard):
        import torch
        from vg_tta.desta3d_v3_data import SourcePool
        from vg_tta.desta3d_v3_joint_mixer import native_supervision
        from vg_tta.desta3d_v3_oracle_mixer_gap import analytic_joint,direction_readout,mixer_readout
        from vg_tta.desta3d_v3_decomposition import shared_fields,norm
        from vg_tta.desta3d_v3_actuation_full_vocab import capture_teacher,replay_branch
        from vg_tta.desta3d_v3_free_actuation import native_ce
        from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_p0 import adapter_sha256
        pr,model,adapter,mixer=setup(cfg['seed']);mixer.eval().requires_grad_(False);basis=mixer.basis.detach()
        pool=SourcePool(load_annotations=False);db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
        rows=read(D/'INPUTS.json');provenance=read(OLD/'VALIDATION_PROVENANCE.json');oldseal=read(E/'PREDICTIONS_SEAL.json')['files']
        completed=new=back=0;start=time.monotonic()
        def frozen():
            assert adapter_sha256(adapter)==ADAPTER_SHA
            assert all(not p.requires_grad and p.grad is None for module in (model,adapter,mixer) for p in module.parameters())
            assert torch.equal(basis.cpu(),load(D/'BASIS.pt'))
        for i,row0 in enumerate(rows):
            guard();ep=D/'episodes'/f'{i:04}'
            if (ep/'COMPLETE.json').exists():
                check_pins({str(ep/p):h for p,h in read(ep/'COMPLETE.json')['files'].items()});completed+=1;continue
            if new>=limit or time.monotonic()-start>cfg['phase_seconds']-120:break
            assert not ep.exists(),'Partial episode requires root review and a new recovery version'
            assert shutil.disk_usage(ROOT).free>cfg['minimum_free_bytes']+2*2**30
            ep.mkdir(parents=True);old_ep=E/'episodes'/f'{i:04}'
            for arm in ARMS[:-1]:
                src=old_ep/(arm+'.pt');assert sha(src)==oldseal[str(src.relative_to(E))];os.link(src,ep/(arm+'.pt'))
            row=json.loads(json.dumps(row0));path,media=pool.media(provenance[row['key']],materialize=True)
            assert media['sha256']==row['input']['video_sha256'];row['input']['video_path']=str(path)
            label=json.loads(db.execute('select labels_json from examples where key=?',(row['key'],)).fetchone()[0])
            prompt,pre,fields,args=example(pr,model,adapter,row,label);stock=fields['visual_grid'].detach()
            support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)};old_input=read(old_ep/'INPUT.json')
            assert support==old_input['support'] and pre==old_input['preprocess'];assert tensor_sha(args[-2])==old_input['evidence_sha']
            write(ep/'INPUT.json',{**old_input,'old_input_sha':sha(old_ep/'INPUT.json'),'source_GT_gradient_oracle':True,'diagnosis_exposed':True})
            with torch.no_grad(),shared_fields(adapter,{'event':stock,'spatial':stock},['event','spatial'],allow_prefix=True) as baseline_calls:
                native,trace=capture_teacher(model,pr,prompt,adapter,fields)
            base=prediction_record(native,row,pre,ADAPTER_SHA);base.update(readout=details(native),support=support,
                GT_read=True,GT_purpose='source gradient diagnosis; baseline itself unmodified',target_read=False)
            torch.save(base,ep/'BASELINE_REPLAY.pt');torch.save(trace,ep/'BASE_TRACE.pt')
            old=load(ep/'B1.pt');keys=['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval','format_ok','preprocess','adapter_sha','event_completion','spatial_completion','readout','video_sha256','support']
            checks={k:equal(base[k],old[k]) for k in keys};write(ep/'BASELINE_CHECK.json',dict(checks=checks,calls=baseline_calls));assert all(checks.values()),checks
            del native,old;gc.collect();gradients={};objectives={}
            for b,j,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
                guard();supervision,missing=native_supervision(label,trace,b)
                if missing:
                    gradients[b]=torch.zeros_like(stock);objectives[b]=dict(missing=missing);continue
                target,valid=supervision;expected=trace['branches'][j]['logits'][kind]
                leaf=stock.detach().clone().requires_grad_();release_free_host_arenas();torch.cuda.empty_cache()
                with shared_fields(adapter,{b:leaf},[b]):
                    logits,cache=replay_branch(model,prompt,adapter,fields,trace,b)
                    exact=torch.equal(logits.detach().cpu(),expected)
                    write(ep/(b+'_FORWARD.json'),dict(exact=exact,classes=logits.shape[-1],actions=int(valid.sum()),native_sha=tensor_sha(expected),replay_sha=tensor_sha(logits.detach()),cache=cache))
                    assert exact,'B1 native/replay mismatch'
                    loss=native_ce(logits,target.cuda(),valid.cuda());loss.backward();back+=1
                assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
                gradients[b]=leaf.grad.detach().clone();objectives[b]=dict(CE=float(loss.detach()),targets=target,valid=valid,classes=logits.shape[-1],actions=int(valid.sum()))
                del leaf,logits,loss,cache;gc.collect();torch.cuda.empty_cache();frozen()
            delta=analytic_joint(gradients['event'],gradients['spatial'],basis,stock,cfg['radius'])
            deltas={'oracle':delta};coeff={};mixer_stats={}
            for arm in SEEDS:
                mixer.load_state_dict(load(OLD/arm/'FINAL.pt')['mixer']);mixer.eval().requires_grad_(False)
                hat,a,meta=mixer_readout(mixer,args);old=load(ep/(arm+'.pt'))
                assert tensor_sha(stock+hat)==old['injection']['corrected_F_sha']
                assert tensor_sha(stock)==old['injection']['common_F_sha']
                assert norm(hat)/norm(stock)==old['injection']['relative_norm']
                deltas[arm]=hat;coeff[arm]=a;mixer_stats[arm]=meta
                del old
            geo=direction_readout(gradients,deltas,stock,cfg['radius']);geo['mixer']=mixer_stats
            raw=dict(gradients={k:v.cpu() for k,v in gradients.items()},deltas={k:v.cpu() for k,v in deltas.items()},coefficients=coeff,objectives=objectives,geometry=geo,stock_shape=list(stock.shape),stock_sha=tensor_sha(stock))
            torch.save(raw,ep/'RAW_DIRECTIONS.pt');write(ep/'GEOMETRY.json',geo)
            corrected=stock+delta
            with torch.no_grad(),shared_fields(adapter,{'event':corrected,'spatial':corrected},['event','spatial'],allow_prefix=True) as calls:
                result=decode_shared_reference_two_pass(model,pr,prompt,adapter,fields)
            pred=prediction_record(result,row,pre,ADAPTER_SHA);pred.update(arm='oracle',readout=details(result),support=support,source_GT_oracle=True,GT_read=True,GT_purpose='source native gradient oracle',decoder_GT_prefix=False,target_read=False,optimizer_steps=0,
                injection=dict(calls=calls,same_field_both_passes=True,common_F_sha=tensor_sha(stock),corrected_F_sha=tensor_sha(corrected),relative_norm=norm(delta)/norm(stock)))
            torch.save(pred,ep/'oracle.pt');validate_prediction(pred,len(row['input']['frame_ids']));frozen()
            files={p.name:sha(p) for p in ep.iterdir() if p.is_file()};write(ep/'COMPLETE.json',dict(index=i,files=files,backwards=sum('missing' not in x for x in objectives.values()),optimizer_steps=0,predictions=4,new_native=2))
            new+=1;completed+=1;print('GAP_COMPLETE',completed,len(rows),'new',new,flush=True)
            del stock,fields,args,prompt,base,trace,gradients,objectives,deltas,delta,hat,a,coeff,raw,corrected,result,pred;gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in (D/'episodes').rglob('*') if p.is_file())<cfg['maximum_new_bytes']
        if completed==len(rows):
            files={str(p.relative_to(D)):sha(p) for p in (D/'episodes').rglob('*') if p.is_file()}
            write(D/'PREDICTIONS_SEAL.json',dict(files=files,predictions=1788,source_GT_diagnosis=True));write(D/'COMPLETE.json',dict(queries=447,parents=31,predictions=1788,seal_sha=sha(D/'PREDICTIONS_SEAL.json'),optimizer_steps=0))
        write(d/'COMPLETE.json',dict(status='complete' if completed==len(rows) else 'safe_pause',completed_queries=completed,new_queries=new,actual_backwards=back,optimizer_steps=0,peak_GPU_allocated=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--name');p.add_argument('--limit',type=int,default=447);a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.name,a.limit)
