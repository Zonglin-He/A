"""A0 Train128/Dev64 cached direction screen; write-once serial stages."""
import argparse,gc,json,os,random,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_joint_learnability import D as OLD,OUT,ROSTER,V1,B1,B1_SHA,ADAPTER_SHA,load,setup,example,equal
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,local_dependencies,allocation,tensor_sha
D=OUT/'a0_fast_screen_v1';GAP=OUT/'oracle_mixer_gap_v1';P=OUT/'gap_followups_v1/decision_readback_v1'
PROTOCOL=ROOT/'protocols/desta3d_v3_a0_fast_screen_v1.md'

def prepare():
    import torch
    from vg_tta.desta3d_v3_a0_screen import select_rows
    from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer
    assert not D.exists()
    result=subprocess.run([sys.executable,'-B','-m','pytest','-q','tests/test_desta3d_v3_a0_screen.py'],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    train,dev=select_rows(read(OLD/'TRAIN_INPUTS.json'),read(GAP/'INPUTS.json'))
    assert len(train)==128 and len({r['source'] for r in train})==95
    assert len(dev)==64 and len({r['source'] for r in dev})==16
    assert not {r['source'] for r in train}&{r['source'] for r in dev}
    ids={r['key']:i for i,r in enumerate(read(GAP/'INPUTS.json'))}
    dev=[{**r,'gap_index':ids[r['key']]} for r in dev]
    cfg=dict(stage='A0_cached_direction_fast_screen',seed=20260928,hidden=128,steps=200,batch=4,
        lr=.001,weight_decay=0,clip=1,radius=read(OLD/'CONFIG.json')['radius'],
        train_queries=128,train_parents=95,dev_queries=64,dev_parents=16,
        phase_seconds=3600,minimum_free_bytes=8*2**30,maximum_new_bytes=32*2**30,cap=None,
        train_cosine_threshold=.3,dev_cosine_threshold=.1,prior_seconds=70591.88224354811,
        target_read=False,fresh_read=False,source_GT=True,loss='query_global_coefficient_cosine_only')
    assert shutil.disk_usage(ROOT).free>cfg['maximum_new_bytes']+cfg['minimum_free_bytes']
    write(D/'CONFIG.json',cfg);write(D/'TRAIN128.json',train);write(D/'DEV64.json',dev)
    write(D/'CPU_PREFLIGHT.json',dict(status='passed',tests=3,GPU=False,output=result.stdout))
    shutil.copyfile(OLD/'BASIS.pt',D/'BASIS.pt');basis=load(D/'BASIS.pt');m=StateAwareDirectionMixer(basis)
    write(D/'SCOPE.json',dict(parameters=sum(p.numel() for p in m.parameters()),trainable_names=[n for n,p in m.named_parameters()],frozen_basis_sha=tensor_sha(basis)))
    paths=local_dependencies([Path(__file__),PROTOCOL,D/'CONFIG.json',D/'TRAIN128.json',D/'DEV64.json',D/'BASIS.pt',D/'CPU_PREFLIGHT.json',
        ROOT/'vg_tta/desta3d_v3_a0_screen.py',ROOT/'tests/test_desta3d_v3_a0_screen.py',ROOT/'vg_tta/desta3d_v3_gap_native_state.py',
        ROOT/'vg_tta/desta3d_v3_gap_candidates.py',V1/'source_fit/SOURCE_TRAIN_RECORDS.json',ROSTER/'SOURCE.sqlite',B1,
        GAP/'PREDICTIONS_SEAL.json',P/'SEAL.json'])
    pins={str(p):sha(p) for p in paths};oldseal=read(GAP/'PREDICTIONS_SEAL.json')['files'];coefseal=read(P/'SEAL.json')
    for r in dev:
        ep=GAP/'episodes'/f"{r['gap_index']:04}"
        for name in ['B1.pt','BASE_TRACE.pt','INPUT.json','GEOMETRY.json']:
            p=ep/name;assert sha(p)==oldseal[str(p.relative_to(GAP))];pins[str(p)]=sha(p)
        p=P/'coefficients'/f"{r['gap_index']:04}"/'ORACLE_COEFFICIENTS.pt';pins[str(p)]=sha(p)
        # The completed coefficient seal is keyed by path relative to P.
        assert pins[str(p)]==coefseal['files'][str(p.relative_to(P))]
    write(D/'LOCK.json',dict(pins=pins));write(D/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',protocol_sha=sha(PROTOCOL),selection='fixed SHA256 metadata only; no results',fresh_read=False))
    print('A0_REGISTERED',cfg,flush=True)

def stage(action,name):
    check_pins(read(D/'LOCK.json')['pins']);d=OUT/name;assert not d.exists()
    cfg=read(D/'CONFIG.json');cfg['phase_seconds']=3600 if action=='cache' else 900
    write(d/'CONFIG.json',cfg);write(d/'LOCK.json',dict(pins={**read(D/'LOCK.json')['pins'],str(D/'LOCK.json'):sha(D/'LOCK.json')}))
    write(d/'REGISTRATION.json',dict(time=time.time(),action=action,source_GT=True,fresh_read=False))
    with allocation(d) as (cfg,guard):
        {'cache':cache,'fit':fit,'native':native}[action](d,cfg,guard)

def labels_for(rows,split):
    if split=='train':
        selected={r['key'] for r in rows}
        return {r['key']:r for r in read(V1/'source_fit/SOURCE_TRAIN_RECORDS.json') if r['key'] in selected}
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
    result={r['key']:json.loads(db.execute('select labels_json from examples where key=?',(r['key'],)).fetchone()[0]) for r in rows};db.close();return result

def cache(d,cfg,guard):
    import torch
    from vg_tta.desta3d_v3_decomposition import shared_fields,norm
    from vg_tta.desta3d_v3_actuation_full_vocab import capture_teacher,replay_branch
    from vg_tta.desta3d_v3_joint_mixer import native_supervision
    from vg_tta.desta3d_v3_oracle_mixer_gap import analytic_joint
    from vg_tta.desta3d_v3_free_actuation import native_ce
    from vg_tta.desta3d_v3_gap_native_state import native_state_from_records
    from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
    from scripts.desta3d_v2_source_fit import prediction_record
    from scripts.desta3d_v2_reference_audit_cached_v3 import details
    from scripts.desta3d_v2_p0 import adapter_sha256
    pr,model,adapter,unused=setup(cfg['seed']);basis=unused.basis.detach();unused.eval().requires_grad_(False)
    backward=0;completed=0
    for split,roster in [('train','TRAIN128.json'),('dev','DEV64.json')]:
        rows=read(D/roster);labels=labels_for(rows,split)
        for i,row in enumerate(rows):
            guard();ep=D/'cache'/split/f'{i:04}';assert not ep.exists(),'No silent replay of partial cache'
            ep.mkdir(parents=True);label=labels[row['key']]
            prompt,pre,fields,args=example(pr,model,adapter,row,label);stock=fields['visual_grid'].detach()
            support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            if split=='train':
                with torch.no_grad(),shared_fields(adapter,{'event':stock,'spatial':stock},['event','spatial'],allow_prefix=True):
                    result,trace=capture_teacher(model,pr,prompt,adapter,fields)
                base=prediction_record(result,row,pre,ADAPTER_SHA);base.update(readout=details(result),support=support)
                torch.save(base,ep/'B1.pt');torch.save(trace,ep/'BASE_TRACE.pt');del result
                gradients={};objectives={}
                for branch,j,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
                    guard();supervision,missing=native_supervision(label,trace,branch)
                    if missing:
                        gradients[branch]=torch.zeros_like(stock);objectives[branch]=dict(missing=missing);continue
                    target,valid=supervision;expected=trace['branches'][j]['logits'][kind]
                    leaf=stock.detach().clone().requires_grad_();release_free_host_arenas();torch.cuda.empty_cache()
                    with shared_fields(adapter,{branch:leaf},[branch]):
                        logits,meta=replay_branch(model,prompt,adapter,fields,trace,branch)
                        exact=torch.equal(logits.detach().cpu(),expected)
                        write(ep/(branch+'_FORWARD.json'),dict(exact=exact,classes=logits.shape[-1],actions=int(valid.sum()),cache=meta));assert exact
                        loss=native_ce(logits,target.cuda(),valid.cuda());loss.backward();backward+=1
                    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
                    gradients[branch]=leaf.grad.detach().clone();objectives[branch]=dict(CE=float(loss.detach()),targets=target,valid=valid,classes=logits.shape[-1])
                    del leaf,logits,loss,meta;gc.collect();torch.cuda.empty_cache()
                delta=analytic_joint(gradients['event'],gradients['spatial'],basis,stock,cfg['radius'])
                oracle=(delta.double()@basis.double()).float().cpu()
                torch.save(dict(gradients={k:v.cpu() for k,v in gradients.items()},objectives=objectives,oracle=oracle,stock_norm=norm(stock)),ep/'ORACLE_RAW.pt')
                del gradients,objectives,delta
            else:
                old=GAP/'episodes'/f"{row['gap_index']:04}";inp=read(old/'INPUT.json')
                assert pre==inp['preprocess'] and support==inp['support'] and tensor_sha(args[-2])==inp['evidence_sha']
                trace=load(old/'BASE_TRACE.pt');base=load(old/'B1.pt');oracle=load(P/'coefficients'/f"{row['gap_index']:04}"/'ORACLE_COEFFICIENTS.pt')
                assert abs(norm(stock)-read(old/'GEOMETRY.json')['stock_norm'])<1e-8
                os.link(old/'B1.pt',ep/'B1.pt')
                write(ep/'REUSE.json',dict(gap_index=row['gap_index'],trace_sha=sha(old/'BASE_TRACE.pt'),oracle_sha=sha(P/'coefficients'/f"{row['gap_index']:04}"/'ORACLE_COEFFICIENTS.pt'),new_oracle_backwards=0,new_native=0,exact_support=True))
            state=native_state_from_records(base,trace,torch.tensor(label['boxes_xyxy'],dtype=torch.float32),torch.tensor(label['box_valid'],dtype=torch.bool)).unsqueeze(0)
            z,qT,qS,ev,_=args
            data=dict(z=z.cpu(),qT=qT.cpu(),qS=qS.cpu(),evidence8=ev.cpu(),state33=state,oracle_coeff256=oracle,stock_norm=norm(stock),stock_shape=list(stock.shape))
            assert oracle.shape==(*z.shape[:-1],256)
            assert all(torch.isfinite(v).all() for v in data.values() if isinstance(v,torch.Tensor))
            torch.save(data,ep/'CACHE.pt');write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],support=support,preprocess=pre,evidence_sha=tensor_sha(ev),state_sha=tensor_sha(state),source_GT=True,target_read=False))
            assert adapter_sha256(adapter)==ADAPTER_SHA and all(p.grad is None and not p.requires_grad for mod in (model,adapter,unused) for p in mod.parameters())
            write(ep/'COMPLETE.json',dict(files={p.name:sha(p) for p in ep.iterdir() if p.is_file()},split=split,index=i,oracle_defined=bool(oracle.norm()>0)))
            completed+=1;print('CACHE_COMPLETE',split,i+1,len(rows),'total',completed,'backward',backward,flush=True)
            del prompt,pre,fields,args,stock,trace,base,oracle,z,qT,qS,ev,state,data;gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in (D/'cache').rglob('*') if p.is_file())<cfg['maximum_new_bytes']
    write(D/'CACHE_SEAL.json',dict(files={str(p.relative_to(D)):sha(p) for p in (D/'cache').rglob('*') if p.is_file()},queries=192,train=128,dev=64))
    write(d/'COMPLETE.json',dict(cache_seal_sha=sha(D/'CACHE_SEAL.json'),queries=192,actual_backwards=backward,new_native=128,optimizer_steps=0,peak_GPU_bytes=torch.cuda.max_memory_allocated()))

def fit(d,cfg,guard):
    import torch,numpy as np
    from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer,coefficient_direction_loss
    from vg_tta.desta3d_v3_a0_screen import coefficients,terminal_decision
    from vg_tta.optimizer_checkpoint import cpu_clone,validate_serialized_optimizer,restore_optimizer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    check_pins({str(D/p):h for p,h in read(D/'CACHE_SEAL.json')['files'].items()})
    m=StateAwareDirectionMixer(load(D/'BASIS.pt'),radius=cfg['radius']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0)
    initial=cpu_clone(m.state_dict());rng=random.Random(cfg['seed']);order=[];hist=[]
    for step in range(1,201):
        guard();opt.zero_grad(set_to_none=True);indices=[];ls=[];valid_count=0
        for j in range(4):
            if not order:order=list(range(128));rng.shuffle(order)
            i=order.pop();indices.append(i);data=load(D/'cache/train'/f'{i:04}'/'CACHE.pt');data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
            a=coefficients(m,data);obj=coefficient_direction_loss(a,data['oracle_coeff256']);(obj['cosine']/4).backward();ls.append(float(obj['cosine'].detach()));valid_count+=int(obj['valid'].sum())
            del data,a,obj
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip']));opt.step()
        counters=[int(s['step']) for s in opt.state.values()];assert set(counters)=={step}
        hist.append(dict(step=step,indices=indices,loss=float(np.mean(ls)),valid=valid_count,gradient_norm=gn,counters=counters))
        if step%20==0:print('A0_STEP',step,'loss',np.mean(ls),flush=True)
    state=cpu_clone(opt.state_dict());validate_serialized_optimizer(state)
    torch.save(dict(mixer=cpu_clone(m.state_dict()),optimizer=state,initial=initial,steps=200,seed=cfg['seed'],cache_seal_sha=sha(D/'CACHE_SEAL.json'),lock_sha=sha(D/'LOCK.json')),D/'FINAL.pt')
    write(D/'HISTORY.json',hist);test=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0);restore_optimizer(test,state)
    assert all(int(v['step'])==200 for v in test.state.values()) and torch.equal(m.basis.cpu(),load(D/'BASIS.pt'))
    results={};m.eval()
    for split,count in [('train',128),('dev',64)]:
        entries=[]
        for i in range(count):
            guard();data=load(D/'cache'/split/f'{i:04}'/'CACHE.pt');data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
            with torch.no_grad():a=coefficients(m,data)
            pred=a.cpu();target=data['oracle_coeff256'].cpu();den=pred.double().norm()*target.double().norm()
            cos=float((pred.double()*target.double()).sum()/den) if den>0 else None
            p=D/'terminal_coefficients'/split/f'{i:04}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(pred,p)
            entries.append(dict(index=i,cosine=cos,sha=sha(p)));del data,a,pred,target
        vals=[x['cosine'] for x in entries if x['cosine'] is not None]
        results[split]=dict(rows=entries,defined=len(vals),undefined=count-len(vals),mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None)
    decision=terminal_decision(results['train']['median'],results['dev']['median'])
    write(D/'DIRECTION_REPORT.json',dict(status='completed_pending_independent_audit',results=results,decision=decision,steps=200,PTD_loaded=False,trainable_parameters=sum(p.numel() for p in m.parameters()),fresh_read=False))
    write(D/'FIT_SEAL.json',dict(files={str(p.relative_to(D)):sha(p) for p in [D/'FINAL.pt',D/'HISTORY.json',D/'DIRECTION_REPORT.json',*(D/'terminal_coefficients').rglob('*.pt')]}))
    write(d/'COMPLETE.json',dict(steps=200,decision=decision,fit_seal_sha=sha(D/'FIT_SEAL.json'),peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    print('A0_DIRECTION_COMPLETE',decision,{s:{k:v for k,v in r.items() if k!='rows'} for s,r in results.items()},flush=True)

def native(d,cfg,guard):
    raise RuntimeError('Native runner requires audited cosine gate; implementation registered separately before use')

if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cache','fit','native']);p.add_argument('--name');a=p.parse_args()
    prepare() if a.action=='prepare' else stage(a.action,a.name)
