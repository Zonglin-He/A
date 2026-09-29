"""One frozen provider, exact loader smoke then16 sealed exposed-source evidence episodes."""
import argparse,sys,os,time,gc,shutil,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,local_dependencies,allocation,total_prior,tensor_sha
from vg_tta.external_qualification_io import CK,SIGLIP,OFFICIAL,DOWNLOAD,seal,verify_seal
D=OUT/'external_policy_gate_v1';A05=OUT/'a05_unlabeled_signal_v1';GAP=OUT/'oracle_mixer_gap_v1'
P=ROOT/'protocols/desta3d_v3_external_policy_gate_v1.md'

def prepare():
    check_pins(read(D/'LOCK.json')['pins'])
    rows=read(D/'INPUTS.json');cfg=read(D/'CONFIG.json')
    extra=local_dependencies([Path(__file__),ROOT/'vg_tta/llava_st_teacher_v2.py',ROOT/'protocols/desta3d_v3_external_policy_loader_repair_v2.md'])
    pins={**read(D/'LOCK.json')['pins'],**{str(p):sha(p) for p in extra}}
    for stage in ['smoke002','evidence002']:
        rd=OUT/('extgate_'+stage);assert not rd.exists()
        write(rd/'CONFIG.json',cfg);write(rd/'INPUTS.json',rows[:2] if stage=='smoke002' else rows)
        write(rd/'LOCK.json',dict(pins={**pins,str(rd/'CONFIG.json'):sha(rd/'CONFIG.json'),str(rd/'INPUTS.json'):sha(rd/'INPUTS.json')}))
        write(rd/'REGISTRATION.json',dict(status='registered_before_GPU',time=time.time(),repair='official AutoConfig instead of Qwen2 default vocab',previous_failure=str(OUT/'extgate_smoke001'),science_unchanged=True))
    print('REGISTERED repair v2',flush=True)

def smoke(run,cfg,guard):
    import torch,numpy as np
    from vg_tta.llava_st_teacher_v2 import load,load_official,predict
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.external_privileged_views import repeated_frame_indices,parse_teacher_text
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    results={};ph={};rows=read(run/'INPUTS.json')
    for loader in ['efficient','official']:
        guard();tok,model,proc,info=load(CK) if loader=='efficient' else load_official(CK,SIGLIP,OFFICIAL)
        ph[loader]={n:tensor_sha(p) for n,p in model.named_parameters()};write(run/(loader+'_LOADING.json'),dict(info=info,parameter_hashes=ph[loader],max_frame=model.config.max_frame))
        results[loader]=[]
        for i,row in enumerate(rows):
            guard();frames,ids=decode(row['input']);bounds=[row['input']['start_frame'],row['input']['end_frame']];ix,_=repeated_frame_indices(ids,clip_bounds=bounds)
            ep=run/'episodes'/loader/f'{i:02}';write(ep/'INPUT.json',dict(key=row['key'],frame_ids=ids,clip_bounds=bounds,pixel_sha256=__import__('hashlib').sha256(frames.tobytes()).hexdigest(),repeat_indices=ix.tolist()))
            recipes={}
            for recipe in ['greedy','official']:
                guard();captured=[]
                def hook(mod,args,result):
                    if not captured:captured.append(result[0].detach().cpu())
                handle=model.get_vision_tower().register_forward_hook(hook)
                try:pred,pixels=predict(tok,model,proc,frames[ix],row['input']['caption'],OFFICIAL,decode=recipe,seed=cfg['seed'])
                finally:handle.remove()
                write(ep/(recipe+'_RAW.json'),pred);assert len(captured)==1 and torch.isfinite(captured[0]).all()
                torch.save(dict(pixels=pixels,first_vision_features=captured[0]),ep/(recipe+'_FEATURES.pt'))
                write(ep/(recipe+'_EVIDENCE.json'),parse_teacher_text(pred['raw_text'],ids,clip_bounds=bounds))
                recipes[recipe]=dict(tokens=pred['output_token_ids'],input_ids=pred['input_token_ids'],pixels_sha=tensor_sha(pixels),vision_sha=tensor_sha(captured[0]))
                print('SMOKE',loader,i,recipe,'TOKENS',len(pred['output_token_ids'][0]),flush=True)
            results[loader].append(recipes)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        del model,tok,proc,pixels,captured;gc.collect();torch.cuda.empty_cache()
    write(run/'COMPARISON.json',dict(results=results,parameters_exact=ph['efficient']==ph['official']))
    assert ph['efficient']==ph['official'] and results['efficient']==results['official'],'Loader not exact'
    same=all(x['greedy']['tokens']==x['official']['tokens'] for x in results['official']);recipe='greedy' if same else 'official'
    write(run/'COMPLETE.json',dict(status='engineering_equivalence_passed',seal_sha=seal(run,8),loader_exact=True,qualification_decode=recipe,decode_same_on_two_inputs=same))

def evidence(run,cfg,guard):
    import torch,numpy as np,hashlib
    from vg_tta.llava_st_teacher_v2 import load,predict
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.external_privileged_views import repeated_frame_indices,parse_teacher_text,dense_spatial_support
    smoke=OUT/'extgate_smoke002';sd,_=verify_seal(smoke);assert sd['loader_exact'];recipe=sd['qualification_decode']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;tok,model,proc,info=load(CK);write(run/'MODEL_LOADING.json',info)
    for i,row in enumerate(read(run/'INPUTS.json')):
        guard();ep=run/'episodes'/f'{i:02}';frames,ids=decode(row['input']);bounds=[row['input']['start_frame'],row['input']['end_frame']];ix,repeated=repeated_frame_indices(ids,clip_bounds=bounds)
        identity=dict(key=row['key'],source=row['source'],frame_ids=ids,clip_bounds=bounds,pixel_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),original_video_sha256=row['input']['video_sha256'],repeated_observation_indices=ix.tolist(),physical_slots=np.linspace(*bounds,100).tolist(),teacher_pixel_frame_ids=repeated.tolist(),frame_shape=list(frames.shape),no_unseen_pixels=True)
        write(ep/'INPUT.json',identity)
        if i<2:
            old=smoke/'episodes/efficient'/f'{i:02}';orig=read(old/'INPUT.json');assert orig['pixel_sha256']==identity['pixel_sha256'] and orig['key']==row['key']
            shutil.copyfile(old/(recipe+'_RAW.json'),ep/'RAW_PREDICTION.json');pred=read(ep/'RAW_PREDICTION.json')
        else:
            pred,pixels=predict(tok,model,proc,frames[ix],row['input']['caption'],OFFICIAL,decode=recipe,seed=cfg['seed']);pred['processed_pixel_sha256']=tensor_sha(pixels);write(ep/'RAW_PREDICTION.json',pred)
        ev=parse_teacher_text(pred['raw_text'],ids,clip_bounds=bounds);_,diag=dense_spatial_support(ids,ev);write(ep/'EVIDENCE.json',ev);write(ep/'SUPPORT.json',diag)
        write(ep/'COMPLETE.json',dict(index=i,from_smoke=i<2,optimizer_steps=0));print('EVIDENCE',i+1,16,'ERRORS',ev['errors'],flush=True)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    write(run/'COMPLETE.json',dict(status='completed_unscored',queries=16,parents=16,predictions=16,decode=recipe,seal_sha=seal(run,16),source_GT_read=False))

def worker(stage):
    run=OUT/('extgate_'+stage)
    with allocation(run) as (cfg,guard):
        (smoke if stage=='smoke002' else evidence)(run,cfg,guard)

def launch(stage):
    run=OUT/('extgate_'+stage);assert not (run/'STARTED.json').exists();check_pins(read(run/'LOCK.json')['pins']);start=time.monotonic()
    env={**os.environ,'PYTHONPATH':':'.join(str(ROOT/p) for p in ['.runtime/llava_st','external/LLaVA-ST','.']),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    cmd=[sys.executable,'-B',str(Path(__file__)),'worker','--stage',stage]
    with (D/(stage+'.log')).open('xb') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);(D/'ACTIVE.json').write_text(json.dumps(dict(status='running',stage=stage,pid=p.pid,wrapper_pid=os.getpid(),time=time.time(),cmd=cmd))+'\n');code=p.wait()
    wall=time.monotonic()-start;receipt=read(run/'RECEIPT.json') if (run/'RECEIPT.json').exists() else dict(seconds=0)
    prior=total_prior();seconds=max(0,wall-receipt['seconds']);write(OUT/('extgate_'+stage+'_wrapper')/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=seconds,worker_seconds=receipt['seconds'],prior_seconds=prior,cumulative_seconds=prior+seconds,cap=None))
    write(run/'EXIT.json',dict(code=code));(D/'ACTIVE.json').write_text(json.dumps(dict(status='completed_pending_audit' if code==0 else 'failed',stage=stage,time=time.time()))+'\n');assert code==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker']);p.add_argument('--stage',choices=['smoke002','evidence002']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='launch':launch(a.stage)
    else:worker(a.stage)
