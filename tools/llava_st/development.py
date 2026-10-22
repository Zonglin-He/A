"""LLaVA-ST backbone: same128 panels, bounded spatial profiles, locked fulls."""
import argparse,copy,gc,json,os,shlex,subprocess,sys,time,fcntl,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stvg.runtime import sha,setup
from stvg.tube_runner import write,digest
from tools.boxfixed_tube.profiles import baseline,profile_id,predict
from tools.llava_st.evidence import SourceEvidence
PUBLIC=ROOT/'results/llava_st_development';OUT=ROOT/'outputs/llava_st_development';PLAN=PUBLIC/'plan.json';PYTHON=Path('/home/wwww/stvg/envs/llava-st/bin/python')


def checked():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    for ds,d in p['datasets'].items():
        for key in ['manifest','full_manifest']:assert sha(ROOT/d[key])==d[key+'_sha256']
    for f,h in p['upstream_code_sha256'].items():assert sha(ROOT/f)==h,f
    return p

def rows_for(ds,full=False):
    p=checked();d=p['datasets'][ds];rows=[json.loads(x)for x in(ROOT/d['full_manifest'if full else'manifest']).read_text().splitlines()];assert len(rows)==(d['full_queries']if full else 128);return p,rows

def model_ready():
    f=ROOT/'runtime/llava_st/checkpoint_seal.json';s=json.loads(f.read_text());assert s['status']=='complete_verified'and s['revision']==checked()['checkpoint_revision']
    for name,e in s['files'].items():assert(ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B'/name).stat().st_size==e['bytes'] and sha(ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B'/name)==e['sha256']
    return s

def setup_model():
    from tools.llava_st.backbone import load
    model_ready();return load()

def native(model,tokenizer,processor,row):
    from stvg.tube_decode import decode
    from stvg.inputs import validate_metadata
    from tools.llava_st.backbone import capture
    validate_metadata(row['metadata']);frames,ids=decode(row['metadata'],row['dataset']);c=capture(model,tokenizer,processor,frames,ids,row['metadata']['caption']);c['request_sha256']=digest(row)
    return c,frames,ids

def qualify():
    setup();p=checked();lock=(ROOT/'runtime/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);model,tok,proc,audit=setup_model();checks=[]
    from stvg.tensors import state_hash
    before=state_hash(model.state_dict())
    from tools.llava_st.teacher_service import Teacher
    teacher=Teacher(OUT/'teacher_cache',p['teacher_fingerprint'])
    for ds in p['dataset_order']:
        _,rows=rows_for(ds);row=rows[0];
        from stvg.tube_decode import decode
        from tools.llava_st.backbone import vision_batch_parity
        parity_frames,_=decode(row['metadata'],row['dataset']);parity=vision_batch_parity(model,proc,parity_frames);del parity_frames
        c,frames,ids=native(model,tok,proc,row);target=OUT/'qualification'/ds;target.mkdir(parents=True,exist_ok=True);path=target/'capture.pt';torch.save(c,path)
        assert c['valid'],(ds,c['reason'],'see private native text in qualification capture')
        assert c['features'].shape[1:]==(81,3584)and len(set(c['positions8']))==8
        c,receipt=get_capture(model,tok,proc,teacher,row,p,cached_native=(c,frames,ids));b,actions,_=decode_spatial(c,p,ds,baseline(ds));assert np.isfinite(b).all()and b.shape==c['Source'].shape;torch.save(c,target/'complete_capture.pt')
        checks.append(dict(dataset=ds,vision_microbatch_parity=parity,previously_evaluated_input=True,native_track_valid=True,frame_nodes=len(c['frame_ids']),generated_track_points=c['generated_track_points'],seconds=c['seconds'],feature_shape=list(c['evidence'].features.shape),capture_sha256=sha(path),complete_capture_sha256=sha(target/'complete_capture.pt'),teacher_queries=receipt['teacher_queries'],labels_read=False))
    assert state_hash(model.state_dict())==before and all(not w.requires_grad and w.grad is None for w in model.parameters())
    write(PUBLIC/'qualification.json',dict(status='pass',checks=checks,model_load=audit,model_weights_unchanged=True,teacher_requests=2,new_teacher_calls=sum(x['teacher_queries']for x in checks),default_BoxFixed_output_finite=True,not_scientific_performance=True,plan_sha256=sha(PLAN)));teacher.close();lock.close()

def get_capture(model,tok,proc,teacher,row,p,cached_native=None):
    from stvg.initialization import initialize
    c,frames,oldids=native(model,tok,proc,row)if cached_native is None else cached_native;receipt=dict(request_sha256=digest(row),labels_read=False,native_valid=c['valid'],reason=c.get('reason'),native_seconds=c['seconds'],seed=c['seed'],new_decoder_generation_calls=1,initialization_scale_provenance='transferred old frozen values; not LLaVA-specific source calibration')
    if c['valid']:
        lookup={x:i for i,x in enumerate(oldids)};teacher_rgb=frames[[lookup[c['frame_ids'][j]]for j in c['positions8']]];e,cost=teacher(teacher_rgb,row['metadata']['caption']);raw,init=initialize(c['Source'].astype(np.float64),c['positions8'],e['boxes'].double().numpy(),p['datasets'][row['dataset']]['configuration']['E_calibration']);evidence=SourceEvidence.from_native_masks(c['features'],c['valid_cells'],c['frame_ids'],c['positions8'],e['mask_logits'],e['teacher_image_size'],c['grid_shape'],selected_query=e['selected_query']);c.update(raw=raw,evidence=evidence,native_boxes=e['boxes'].double(),initialization=init)
        receipt.update(teacher_queries=0 if cost['cache_hit']else 1,teacher_cache_hit=cost['cache_hit'],teacher_cache_key=cost['cache_key'],teacher_seconds=cost['seconds'],positions8=c['positions8'])
    else:receipt.update(teacher_queries=0,invalid_native_output_retained=True)
    del frames;c.pop('features');c.pop('valid_cells');return c,receipt

def decode_spatial(c,p,ds,params):
    if not c['valid']:return np.array(c['Source']),np.zeros(len(c['frame_ids']),dtype=np.int64),None
    from tools.llava_st.spatial import build_geometry
    from stvg.predictor import output_writeback
    E=torch.from_numpy(output_writeback(c['Source'],c['raw'],baseline(ds)['gamma']));g=build_geometry(c['evidence'],E,c['native_boxes']);r=predict(g,c['evidence'],c['Source'],c['raw'],c['Native_interval'],params)
    return r['final'].numpy(),r['actions'].numpy(),r.get('candidate_admitted_frames')

def save_prediction(path,c,boxes,actions):
    np.savez(path,Source=c['Source'],Ours=boxes,frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c.get('positions8',[]),dtype=np.int64),Native_interval=np.asarray(c['Native_interval']),actions=actions,native_valid=np.asarray(c['valid']))

def gpu(ds,full=False):
    setup();p,rows=rows_for(ds,full);assert json.loads((PUBLIC/'qualification.json').read_text())['status']=='pass';lock=(ROOT/'runtime/gpu.lock').open('a');print('Waiting for shared gpu.lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
    model,tok,proc,load_audit=setup_model()
    from tools.llava_st.teacher_service import Teacher
    teacher=Teacher(OUT/'teacher_cache',p['teacher_fingerprint']);target=OUT/ds/('full'if full else'captures');target.mkdir(parents=True,exist_ok=True);entries={}
    if full:
        locks=json.loads((PUBLIC/'selected_profiles.json').read_text());assert set(locks['datasets'])==set(p['dataset_order'])and locks['plan_sha256']==sha(PLAN);selected=locks['datasets'][ds];params=selected['parameters'];devroot=OUT/ds/'profiles'/selected['profile_id'];devseal=json.loads((devroot/'sealed.json').read_text());assert sha(devroot/'sealed.json')==selected['prediction_seal_sha256']
    try:
        for i,row in enumerate(rows):
            q=row['id'];path=target/(q+('.npz'if full else'.pt'));rp=path.with_suffix('.json')
            if path.exists()and rp.exists():
                old=json.loads(rp.read_text());assert old['plan_sha256']==sha(PLAN)and old['request_sha256']==digest(row)and old['payload_sha256']==sha(path)
                if full:assert old['selected_profile_id']==selected['profile_id']
            else:
                if full and q in devseal['files']:
                    e=devseal['files'][q];assert sha(devroot/e['path'])==e['sha256'];receipt=json.loads((devroot/e['receipt_path']).read_text());assert receipt['request_sha256']==digest(row);shutil.copy2(devroot/e['path'],path);receipt.update(reused_development=True,selected_profile_id=selected['profile_id'])
                else:
                    c,receipt=get_capture(model,tok,proc,teacher,row,p)
                    if full:
                        b,actions,admitted=decode_spatial(c,p,ds,params);save_prediction(path,c,b,actions);receipt.update(reused_development=False,selected_profile_id=selected['profile_id'],parameters=params,admitted_frames=admitted)
                    else:torch.save(c,path)
                receipt.update(plan_sha256=sha(PLAN),request_sha256=digest(row),payload_sha256=sha(path));write(rp,receipt)
            entries[q]=dict(path=path.name,sha256=sha(path),receipt_path=rp.name,receipt_sha256=sha(rp));write(OUT/'worker_progress.json',dict(stage='full'if full else'capture128',dataset=ds,complete=i+1,total=len(rows),labels_read=False));print(ds,i+1,'/',len(rows),flush=True)
        assert all(not w.requires_grad and w.grad is None for w in model.parameters());write(target/'sealed.json',dict(status='complete',queries=len(rows),files=entries,plan_sha256=sha(PLAN),manifest_sha256=p['datasets'][ds]['full_manifest_sha256'if full else'manifest_sha256'],labels_read=False,selected_profile=selected if full else None))
    finally:teacher.close();lock.close()

def verify(root):
    s=json.loads((root/'sealed.json').read_text());assert s['status']=='complete'and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(root/e['path'])==e['sha256']and sha(root/e['receipt_path'])==e['receipt_sha256']
    return s

def profile(ds,request):
    setup();p,_=rows_for(ds);params=json.loads(Path(request).read_text());caps=OUT/ds/'captures';s=verify(caps);assert s['queries']==128;target=OUT/ds/'profiles'/profile_id(params);target.mkdir(parents=True,exist_ok=True);entries={}
    for i,(q,e)in enumerate(s['files'].items()):
        c=torch.load(caps/e['path'],map_location='cpu',weights_only=False);b,actions,admitted=decode_spatial(c,p,ds,params);path=target/(q+'.npz');save_prediction(path,c,b,actions);receipt=dict(plan_sha256=sha(PLAN),parameters=params,request_sha256=c['request_sha256'],capture_sha256=e['sha256'],native_valid=c['valid'],reason=c.get('reason'),labels_read=False,new_neural_calls=0,admitted_frames=admitted);write(path.with_suffix('.json'),receipt);entries[q]=dict(path=path.name,sha256=sha(path),receipt_path=path.with_suffix('.json').name,receipt_sha256=sha(path.with_suffix('.json')));write(OUT/'worker_progress.json',dict(stage='profile',dataset=ds,complete=i+1,total=128,parameters=params,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,parameters=params,files=entries,plan_sha256=sha(PLAN),capture_seal_sha256=sha(caps/'sealed.json'),labels_read=False))

def score(ds,request=None,full=False):
    p,rows=rows_for(ds,full);params=json.loads(Path(request).read_text())if request else None;root=OUT/ds/('full'if full else'profiles/'+profile_id(params));s=verify(root);assert s['queries']==len(rows)
    d=p['datasets'][ds];key='full_labels'if full else'labels';assert sha(ROOT/d[key])==d[key+'_sha256'];truth={x['id']:x for x in map(json.loads,(ROOT/d[key]).read_text().splitlines())}
    from stvg.metrics import DenseTube
    details=[];invalid=0
    for row in rows:
        q=row['id'];gt=truth[q];boxes={int(k):v for k,v in gt['boxes'].items()};e=s['files'][q]
        with np.load(root/e['path'])as z:
            ids=z['frame_ids'];iv=z['Native_interval'];valid=bool(z['native_valid']);invalid+=not valid;meta=dict(input=row['metadata'],frame_ids=ids.tolist());obs=set(ids[z['positions8']].tolist());values={};unobserved={};observed={}
            for name in ['Source','Ours']:
                if valid:
                    m=DenseTube(z[name],meta,boxes,gt['interval'],clip=ds=='hc2');values[name]=m.score(iv);mask=np.array([int(t)not in obs for t in m.fids]);unobserved[name]=float(m.iou[mask].mean())if mask.any()else None;observed[name]=float(m.iou[~mask].mean())if(~mask).any()else None
                else:
                    t=max(0,min(int(iv[1]),gt['interval'][1])-max(int(iv[0]),gt['interval'][0]))if iv[1]>iv[0]else 0;union=int(iv[1]-iv[0])+gt['interval'][1]-gt['interval'][0]-t;values[name]=dict(v=0.,s=0.,t=t/union if iv[1]>iv[0]else 0.);unobserved[name]=0.;observed[name]=None
        details.append(dict(id=q,parent=gt['parent'],native_valid=valid,values=values,unobserved=unobserved,observed=observed))
    mean=lambda xs:float(np.mean([x for x in xs if x is not None]))if any(x is not None for x in xs)else None
    result=dict(status='complete',dataset=ds,queries=len(rows),parameters=params,means={a:{k:100*mean([x['values'][a][k]for x in details])for k in ['v','s','t']}for a in ['Source','Ours']},unobserved_sIoU={a:100*mean([x['unobserved'][a]for x in details])for a in ['Source','Ours']},observed_sIoU={a:100*mean([x['observed'][a]for x in details])if mean([x['observed'][a]for x in details])is not None else None for a in ['Source','Ours']},invalid_native_outputs_retained=invalid,metric_denominators={a:dict(main=len(details),unobserved=sum(x['unobserved'][a]is not None for x in details),observed=sum(x['observed'][a]is not None for x in details))for a in ['Source','Ours']},loss_over5pp_queries=sum(x['values']['Ours']['v']<x['values']['Source']['v']-.05 for x in details),prediction_seal_sha256=sha(root/'sealed.json'),fixed_native_time=True,no_independent_unexposed_test_claim=True)
    write(root/'details_private.json',details);write(root/'metrics.json',result)
    if full:
        from tools.tubedetr_opd_direct_diagnosis import cluster_difference
        result['paired']={k:cluster_difference([x['unobserved']['Ours']-x['unobserved']['Source']if k=='unobserved'else x['values']['Ours'][k]-x['values']['Source'][k]for x in details],[x['parent']for x in details])for k in ['s','v','unobserved']};write(PUBLIC/ds/'full_metrics.json',result);(PUBLIC/ds/'report.md').write_text('# LLaVA-ST backbone: complete target population\n\nSame merged checkpoint on both domains, no strict source-only cross-domain claim. Source/Ours share native generated time, no Global6/MBR. Exposed128 development included in full. Invalid outputs retained, never replace query.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n');write(PUBLIC/ds/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/ds/'selected_profile.json',PUBLIC/ds/'all_metrics.json',PUBLIC/ds/'decisions.json',PUBLIC/ds/'full_metrics.json',PUBLIC/ds/'report.md']},prediction_seal_sha256=sha(root/'sealed.json'),publication_pending=True))

def child(stage,ds=None,request=None):
    cmd=[str(PYTHON),'-B','-u',str(Path(__file__).relative_to(ROOT)),'--stage',stage]
    if ds:cmd+=['--dataset',ds]
    if request:cmd+=['--request',str(request)]
    log=ROOT/'logs'/f'llava_st_{ds or "common"}_{stage}.log';write(OUT/'status.json',dict(stage=stage,dataset=ds,request=str(request)))
    with log.open('a')as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''}if stage in ['profile','score','full_score']else os.environ.copy());jobs=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(jobs.read_text());j['jobs']['llava_st_controller']=dict(pid=os.getpid(),command='tools/llava_st/development.py --stage run',log='logs/complete_llava_st_development.log');j['jobs']['llava_st_worker']=dict(pid=proc.pid,command=shlex.join(cmd),stage=stage,dataset=ds,log=str(log.relative_to(ROOT)));write(jobs,j)
        if proc.wait():raise RuntimeError(f'{stage} {ds} failed; see {log}')

def run():
    p=checked();OUT.mkdir(parents=True,exist_ok=True)
    registry=ROOT/'runtime/tubedetr_active_jobs.json';jobs=json.loads(registry.read_text());jobs['jobs']['llava_st_controller']=dict(pid=os.getpid(),command=str(PYTHON)+' -B -u tools/llava_st/development.py --stage run',log='logs/complete_llava_st_development.log');write(registry,jobs)
    while not(ROOT/'runtime/llava_st/checkpoint_seal.json').exists():
        failure=ROOT/'runtime/llava_st/download_failure.json'
        if failure.exists():raise RuntimeError('Checkpoint download failed; inspect '+str(failure))
        status=ROOT/'runtime/llava_st/download_status.json'
        if status.exists():
            download=json.loads(status.read_text());pid=download.get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if pid and(not proc.exists()or b'tools/llava_st/download_assets.py'not in proc.read_bytes()):raise RuntimeError('Download process exited or identity changed without verified checkpoint seal')
        write(OUT/'status.json',dict(stage='waiting_for_verified_checkpoint',GPU_work_started=False,controller_pid=os.getpid()));time.sleep(30)
    child('qualify');selected={}
    for ds in p['dataset_order']:
        child('capture',ds);tested={};decisions=[];pub=PUBLIC/ds;pub.mkdir(parents=True,exist_ok=True)
        def evaluate(params):
            key=profile_id(params)
            if key not in tested:
                assert len(tested)<24;req=OUT/ds/'requests'/(key+'.json');write(req,params);child('profile',ds,req);child('score',ds,req);tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text());write(pub/'all_metrics.json',list(tested.values()))
            return tested[key]
        current=baseline(ds);evaluate(current)
        for stage,key,grid in [('initialization','gamma',p['gamma_grid']),('admission','admission',p['admission_grid']),('protection','guard',p['guard_grid']),('association','link',p['link_grid'])]:
            candidates=[evaluate(dict(current,**{key:x}))for x in grid];candidates.append(evaluate(current));best=max(candidates,key=lambda m:(m['means']['Ours']['v'],m['parameters']==current));current=best['parameters'];decisions.append(dict(stage=stage,reason_before='Same finite sequential spatial search as prior protocol; native time frozen, all128 queries retained',candidates=candidates,selected=current));write(pub/'decisions.json',decisions)
        i=p['gamma_grid'].index(current['gamma']);neighbors=[p['gamma_grid'][j]for j in [i-1,i+1]if 0<=j<len(p['gamma_grid'])];candidates=[evaluate(dict(current,gamma=x))for x in neighbors]+[evaluate(current)];current=max(candidates,key=lambda m:(m['means']['Ours']['v'],m['parameters']==current))['parameters'];win=evaluate(current);decisions.append(dict(stage='at_most_two_neighbor_gammas',candidates=candidates,selected=current));write(pub/'decisions.json',decisions)
        lock=dict(status='locked_before_full',parameters=current,profile_id=profile_id(current),metrics=win,profiles=len(tested),plan_sha256=sha(PLAN),prediction_seal_sha256=win['prediction_seal_sha256']);write(pub/'selected_profile.json',lock);selected[ds]=lock
    write(PUBLIC/'selected_profiles.json',dict(status='both_domains_locked_before_any_full',datasets=selected,plan_sha256=sha(PLAN)))
    for ds in p['dataset_order']:child('full',ds);child('full_score',ds)
    files=[PLAN,PUBLIC/'qualification.json',PUBLIC/'selected_profiles.json',*[PUBLIC/ds/'final_seal.json'for ds in p['dataset_order']]];write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True));write(OUT/'status.json',dict(status='complete_both128_and_full',full_counts={ds:p['datasets'][ds]['full_queries']for ds in p['dataset_order']}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['run','qualify','capture','profile','score','full','full_score']);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
    try:
        if args.stage=='run':run()
        elif args.stage=='qualify':qualify()
        elif args.stage=='capture':gpu(args.dataset)
        elif args.stage=='profile':profile(args.dataset,args.request)
        elif args.stage=='score':score(args.dataset,args.request)
        elif args.stage=='full':gpu(args.dataset,True)
        else:score(args.dataset,full=True)
    except Exception as e:write(OUT/'failure.json',dict(stage=args.stage,dataset=args.dataset,error=type(e).__name__,message=str(e)));raise
