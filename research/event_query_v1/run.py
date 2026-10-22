"""HC64 sealed evidence and true temporal-query fits, separate from paused full."""
import os,sys,json,time,hashlib,subprocess,traceback,gc,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(Path(__file__).parent))
D=R/'outputs/event_query_hc64_v1';P=R/'results/event_query_hc64_v1';B=R.parent/'outputs/tube_temporal_hc64'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def records():return [json.loads(s)for s in(R/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()and json.loads(s)['dataset']=='hc2'][:64]
def verify():
    plan=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
    for name,h in plan['code'].items():assert sha(R/name)==h,name
    for key,e in plan['captures'].items():
        assert sha(B/'captures'/f'{key}.pt')==e['capture_sha256']
    return plan
def prepare():
    assert not(D/'plan.json').exists();P.mkdir(parents=True,exist_ok=True)
    lk=read(B/'lock.json');seal=read(B/'sealed.json');rows=records();assert len(rows)==64 and [r['id']for r in rows]==lk['ids'];assert sha(R/'data/evaluation/development.jsonl')==lk['manifest_sha256']
    captures={}
    for row in rows:
        k=row['id'];rp=B/'receipts'/f'{k}.json';e=seal['files'][k];assert sha(rp)==e['receipt_sha256'];rec=read(rp);assert sha(B/'captures'/f'{k}.pt')==rec['capture_sha256'];assert sha(B/'predictions'/f'{k}.npz')==e['sha256'];captures[k]=dict(capture_sha256=rec['capture_sha256'],receipt_sha256=sha(rp),prediction_sha256=e['sha256'])
    code={str(p.relative_to(R)):sha(p)for p in sorted(Path(__file__).parent.glob('*.py'))}
    for rel in ['stvg/temporal.py','stvg/global6.py','stvg/temporal_hidden.py','stvg/teacher.py','stvg/source_loader.py','stvg/runtime.py','stvg/spatial.py','stvg/_spatial_math.py']:code[rel]=sha(R/rel)
    plan=dict(schema='event_query_hc64_v1',authorization='User: pause full; experimentally inspect old token misranking and learn real temporal query; same spatial branch',count=64,dataset='hc2',exposure='All64 previously exposed development, not independent confirmation',code=code,captures=captures,manifest_sha256=sha(R/'data/evaluation/development.jsonl'),source_checkpoint_sha256=sha(R/'checkpoints/TASTVG_VidSTG.pth'),teacher_checkpoint_sha256=sha(R/'checkpoints/referdino/ryt_mevis_swinb.pth'),arms=['NativeG6','LegacyContrast','RawDirect','RawQuery','PredictiveDirect','PredictiveQuery'],feedback='same original event-word/subword balanced pre-max curve, whole-track association via E at original8; raw contrast or LOO ridge1 nonnegative effect prediction; prior q0 times exp(score), unit power, unknown score0',query='disposable256 vector, shared both offsets; delta=native-query RMS * u; Adam lr.01 10 steps, normalized residual L2 radius.1, fresh optimizer, last step; same teacher as Direct; no GT selection',steps=10,lr=.01,radius=.1,seed=20261022,expert_positions=8,additional_expert_model=0,maximum_new_expert_queries=64,new_Source_encoder=0,new_network_training=False,spatial='fixed current complete-box geometry, exact DP on each new interval; fixed-space analysis uses NativeG6 path',GPU_budget_seconds=1200,all_predictions_before_new_GT=True,automatic_full_resume=False)
    plan['labels_sha256']=sha(R/'data/evaluation/development_labels.jsonl')
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();put(D/'plan.json',plan)
    put(P/'protocol.json',{k:v for k,v in plan.items()if k not in ['captures']});put(D/'state.json',dict(status='prepared',fingerprint=plan['fingerprint']))
    for rel in code:
        f=D/'source_snapshot'/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/rel).read_bytes())
    print(json.dumps(dict(status='prepared',fingerprint=plan['fingerprint'],queries=64)))
def semantic():
    import numpy as np,torch
    from stvg.runtime import setup,lease
    from stvg.decode import decode
    from stvg.query import QuerySubjectParser
    from legacy_event import event_words,word_tokens,semantic_curve,associate
    p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='semantic_running',**who(),fingerprint=p['fingerprint']))
    from stvg.teacher import SingleRefer
    teacher=SingleRefer();model=teacher.model;parser=QuerySubjectParser(R/'checkpoints/stanza');files={};calls=0
    try:
        for row in records():
            assert time.perf_counter()-start<p['GPU_budget_seconds'];key=row['id'];f=D/'semantics'/f'{key}.pt';assert not f.exists(),'Never overwrite'
            c=torch.load(B/'captures'/f'{key}.pt',map_location='cpu',weights_only=False);meta=row['metadata'];frames,ids=decode(meta,'hc2');positions=list(c['planned_positions']);assert ids==c['frame_ids'];captured=[]
            def hook(module,args,out):captured.append((out.detach().cpu(),out.max(-1,keepdim=True).values.sigmoid().detach().cpu()))
            handle=model.class_embed[-1].register_forward_hook(hook);tick=time.perf_counter();original_infer=model.infer;full_output=[]
            def retain(*args,**kwargs):
                output=original_infer(*args,**kwargs);full_output.append((output['pred_boxes'][0].detach().cpu(),output['pred_logits'][0].detach().cpu()));return output
            model.infer=retain
            try:result=teacher(frames[positions],meta['caption']);calls+=1
            finally:handle.remove();model.infer=original_infer
            # Last call is the actual tracked, temporally fused classification.
            logits,original_sigmoid=captured[-1];assert logits.shape[0]==8;maximum=logits.max(-1,keepdim=True).values
            assert torch.equal(maximum,full_output[0][1]),'Premax must recover exact native logits'
            native_idx=result['selected_query'];assert torch.equal(original_sigmoid[:,native_idx].float(),result['selected_frame_scores'])
            assert torch.equal(result['boxes'],c['geometry']['native_boxes'].float()),'Same8 native expert evidence did not restore'
            idx,association=associate(c['E'][positions],full_output[0][0].numpy())
            text=' '.join(meta['caption'].lower().split());tokens=model.tokenizer([text+'.'if not text.endswith('.')else text],return_offsets_mapping=True)['offset_mapping'][0]
            doc=parser.nlp(meta['caption']);words=[]
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    for w in token.words:words.append(dict(id=w.id,head=w.head,upos=w.upos,deprel=w.deprel,text=w.text,start=token.start_char,end=token.end_char))
            groups,reason=word_tokens(meta['caption'],event_words(words),tokens)
            # Capture all tracked boxes without changing original inference.
            # Original selected track is fixed; no per-frame identity switching.
            curve=semantic_curve(logits.numpy(),groups,idx)
            payload=dict(curve=curve,event_status=reason,groups=groups,premax=logits,selected_query=idx,native_selected_query=native_idx,association=association,native_boxes=result['boxes'],native_scores=result['selected_frame_scores'],positions8=positions,frame_ids=ids,caption_sha256=result['text_sha256'],capture_sha256=p['captures'][key]['capture_sha256'],cost_seconds=time.perf_counter()-tick,GT_read=False)
            f.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,f);files[key]=dict(path=str(f.relative_to(D)),sha256=sha(f),event_status=reason)
            put(D/'progress.json',dict(stage='semantics',completed=len(files),total=64,seconds=time.perf_counter()-start,**who()));print('semantic',key,round(time.perf_counter()-start,2),flush=True);del c,frames,payload,logits,captured;gc.collect();torch.cuda.empty_cache()
        put(D/'semantics/sealed.json',dict(files=files,count=64,new_expert_queries=calls,seconds=time.perf_counter()-start,GT_read=False,plan_sha256=sha(D/'plan.json')))
    finally:lock.close()
def fit():
    import numpy as np,torch
    from stvg.runtime import setup,lease,load_source
    from stvg.temporal_hidden import verify_decoder
    from stvg.global6 import posterior
    from stvg.temporal import final_mbr
    from stvg.spatial import SpatialGeometry,evidence_dp
    from core import restore_inputs,score_feedback,target,torch_posterior
    from legacy_event import candidates,choose
    p=verify();ss=read(D/'semantics/sealed.json');assert ss['count']==64;setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='fit_running',fingerprint=p['fingerprint'],**who()))
    cfg=read(R/'configs/default.json')['datasets']['hc2'];model=load_source(cfg);decoder=model.ground_decoder.time_decoder;head=model.temp_embed;verify_decoder(decoder);model.eval().requires_grad_(False);files={};maxmath=0.;steps=0
    try:
        for row in records():
            assert time.perf_counter()-start<p['GPU_budget_seconds'];key=row['id'];cap=torch.load(B/'captures'/f'{key}.pt',map_location='cpu',weights_only=False);e=ss['files'][key];assert sha(D/e['path'])==e['sha256'];sem=torch.load(D/e['path'],map_location='cpu',weights_only=False)
            offsets=cap['temporal_cache']['offsets'];kw=[restore_inputs(v,'cuda')for v in offsets];ids=cap['frame_ids'];positions=cap['planned_positions'];records_=cap['records'];tick=time.perf_counter();logs=[]
            with torch.no_grad():
                for k,v in zip(kw,offsets):
                    hidden=decoder(**k);assert torch.equal(hidden.cpu(),v['hidden']),'Native hidden bitwise restore failed';z=head(hidden);assert torch.equal(z[-1].cpu(),v['native_logits']);logs.append(z)
            q0=posterior(records_,[v.detach().cpu()for v in logs],ids);qt=torch_posterior(records_,logs,ids);err=float(np.max(abs(qt.detach().cpu().numpy()-q0)));assert err<2e-12;maxmath=max(maxmath,err)
            baseline=final_mbr(ids,q0)['interval'];pairs,raw,predict,feedback_meta=score_feedback(sem['curve'],ids,positions);targets=dict(Raw=target(q0,pairs,raw),Predictive=target(q0,pairs,predict));iv={'NativeG6':baseline}
            edges,cs=candidates(ids,positions,row['metadata']['frame_count'],baseline);iv['LegacyContrast']=choose(sem['curve'],edges,cs,baseline)[0];paths={};saved_q={'NativeG6':q0}
            scale=float(torch.stack([k['query_tgt'][0,0].square().mean().sqrt()for k in kw]).mean());assert scale>0
            for kind,t in targets.items():
                iv[kind+'Direct']=final_mbr(ids,t)['interval'];saved_q[kind+'Direct']=t;u=torch.zeros(256,device='cuda',requires_grad=True);opt=torch.optim.Adam([u],lr=p['lr']);path=[];tt=torch.as_tensor(t,device='cuda',dtype=torch.float64)
                for step in range(p['steps']+1):
                    z=[]
                    for k in kw:
                        new=dict(k);base=k['query_tgt'];assert torch.equal(base,base[:1].expand_as(base))
                        new['query_tgt']=(base[:1]+scale*u[None,None,:]).expand_as(base)
                        z.append(head(decoder(**new)))
                    q=torch_posterior(records_,z,ids);loss=-(tt*torch.log(q.clamp_min(1e-300))).sum();assert torch.isfinite(loss)
                    grad=None
                    if step<p['steps']:
                        opt.zero_grad();loss.backward();grad=float(u.grad.norm());assert np.isfinite(grad);opt.step();steps+=1
                        with torch.no_grad():u.mul_(min(1.,p['radius']*16/max(float(u.norm()),1e-30)))
                    path.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad,normalized_delta_RMS=float(u.detach().norm()/16)))
                    if step==0:assert np.max(abs(q.detach().cpu().numpy()-q0))<2e-12,'Query zero state must preserve posterior'
                    if step==p['steps']:
                        final=posterior(records_,[v.detach().cpu()for v in z],ids);iv[kind+'Query']=final_mbr(ids,final)['interval'];saved_q[kind+'Query']=final
                    del z,q,loss
                paths[kind]=dict(path=path,delta=(scale*u.detach()).cpu(),scale=scale,optimizer_steps=p['steps']);del u,opt
            geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={};arrpaths={}
            for arm,span in iv.items():
                selected=evidence_dp(geom.for_interval(span));arrays[arm]=selected['final'].numpy();arrpaths[arm]=selected.get('path')
            reference=B/'predictions'/f'{key}.npz';assert sha(reference)==p['captures'][key]['prediction_sha256']
            with np.load(reference,allow_pickle=False)as old:source_interval=old['B_Native224_interval'].copy()
            out=D/'predictions'/f'{key}.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**{**arrays,**{a+'_interval':v for a,v in iv.items()},'Source':cap['Source'],'Source_interval':source_interval,'frame_ids':ids,'positions8':positions})
            trace=D/'traces'/f'{key}.pt';trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(paths=paths,q=saved_q,curve=sem['curve'],feedback=dict(pairs=pairs,raw=raw,predictive=predict,metadata=feedback_meta),native_bitwise=True,posterior_math_error=err,GT_read=False),trace)
            files[key]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),seconds=time.perf_counter()-tick)
            put(D/'progress.json',dict(stage='fits',completed=len(files),total=64,seconds=time.perf_counter()-start,**who()));print('fit',key,round(time.perf_counter()-tick,3),flush=True);del cap,sem,kw,logs,qt,paths,arrays,geom;gc.collect();torch.cuda.empty_cache()
        assert all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(files=files,count=64,plan_sha256=sha(D/'plan.json'),semantics_seal_sha256=sha(D/'semantics/sealed.json'),posterior_math_max_error=maxmath,actual_optimizer_steps=steps,network_weights_frozen=True,GT_read=False,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
    finally:lock.close()
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        D.mkdir(exist_ok=True,parents=True);put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True));raise
