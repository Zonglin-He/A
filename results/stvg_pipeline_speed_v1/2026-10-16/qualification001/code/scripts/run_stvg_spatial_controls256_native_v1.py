"""Frozen WHO evidence policies on one locked256 cohort; no outcome tuning."""
from pathlib import Path
import copy,gc,hashlib,json,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha,guard
from scripts.run_stvg_spatial_who_order_v1 import ordered,parse
UNIT='stvg-spatial-controls256-native-001.service';ARMS=('Full_E1','A','B','C','FULL_ORDER','A_ORDER')
def verify():
    for p,h in read(BASE/'NATIVE_DISPATCH_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert read(BASE/'SPATIAL_INPUT_HANDOFF.json')['status']=='complete_pending_native_evidence'
def load():
    from scripts import run_stvg_referent_abc002 as old
    old.BASE=BASE;return old.load_teacher()
def job_for(j,arm):
    x=copy.deepcopy(j)
    if arm in ('FULL_ORDER','A_ORDER'):x['prompt']=ordered(x['prompt']);x['prompt_sha256']=hashlib.sha256(x['prompt'].encode()).hexdigest()
    return x
def prepare(m,p,device,j,frozen=None):
    import torch,numpy as np
    from scripts.stvg_referent_abc_native002 import processor_batch,pack,geometry
    from scripts.stvg_referent_abc_tokens002 import select as choose
    b,ps=processor_batch(p,j,ROOT,device);dims=read(BASE/'SOURCE_GEOMETRY.json')['records'][j['case_id']];v,ma,gr=geometry(j,dims['width'],dims['height'])
    with torch.inference_mode():e,pos,delta,f,runs=pack(m,b,torch.from_numpy(frozen).to(device=device,dtype=m.dtype)if frozen is not None else None)
    qtext=read(BASE/'temporal/requests'/(j['case_id']+'.json'))['query_text'];cap=next(x for x in read(BASE/'CAPTION_LOCK.json')['records']if x['case_id']==j['case_id'])
    def te(s):
        ids=p.tokenizer(s,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
        with torch.inference_mode():return m.get_input_embeddings()(ids)[0].float().cpu().numpy()
    q=te(qtext)
    if cap['available']:assert sha(ROOT/cap['path'])==cap['sha256'];c=te(read(ROOT/cap['path'])['raw_caption'])
    else:c=q.copy() # Never used to make B/C feedback; A/Full are caption-independent.
    z=f.float().cpu().numpy();pol=[choose(z[a:d].reshape(len(v[i]),256,-1),q,c,v[i],ma[i],gr[i])for i,(a,d)in enumerate(((0,512),(512,1280)))];return b,ps,e,pos,f,runs,pol,v,ma,gr,q,c,cap['available']
def cache_write(j,proof,ps,f,pol,v,ma,gr,q,c,available):
    import numpy as np
    b=BASE/'native_cache'/j['job_id'];b.mkdir(parents=True,exist_ok=False);np.savez_compressed(b/'native.npz',features=f.float().cpu().numpy(),query_embedding=q,caption_embedding=c,**{f'valid{i}':x for i,x in enumerate(v)},**{f'mandatory{i}':x for i,x in enumerate(ma)},**{f'groups{i}':x for i,x in enumerate(gr)});serial=[{k:({a:v.tolist()for a,v in z.items()}if k=='indices'else z)for k,z in r.items()}for r in pol];put(b/'POLICIES.json',dict(status='locked',records=serial,job_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),prompt_sha256=ps,native_sha256=sha(b/'native.npz'),model_clock_proof=proof,caption_available=available,GT_read=False))
def qualification():
    import torch
    from scripts.stvg_referent_abc_native002 import sequence_keep,greedy,processor_batch,pack
    verify();sys.addaudithook(guard);m,p,device,proof,secs=load();rows=[]
    for ds in ('hc2','vidstg'):
        j=next(j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']if j['dataset']==ds);b,ps,e,pos,f,runs,pol,v,ma,gr,q,c,available=prepare(m,p,device,j);cache_write(j,proof,ps,f,pol,v,ma,gr,q,c,available)
        for version in ('old','ordered'):
            if version=='ordered':b,ps=processor_batch(p,job_for(j,'FULL_ORDER'),ROOT,device)
            with torch.inference_mode():
                e,pos,_,f,runs=pack(m,b,f);m.model.rope_deltas=None;old=m.generate(**b,max_new_tokens=8,do_sample=False,use_cache=True)[0,b['input_ids'].shape[1]:].cpu().tolist();ours,_=greedy(m,e,pos,sequence_keep(b,m,runs,[range(512),range(768)]),b['input_ids'],8);assert old==ours
                for arm in (('A','B','C')if version=='old'else('A',)):
                    _,tr=greedy(m,e,pos,sequence_keep(b,m,runs,[x['indices'][arm]for x in pol]),b['input_ids'],8,check_recompute=True);assert tr['cached_uncached_next_token_checks']>=2
            rows.append(dict(dataset=ds,prompt_version=version,native_first8_exact=True,pruned_cached_uncached_checks=True,GT_read=False))
        del b,e,pos,f,runs,pol;gc.collect();torch.cuda.empty_cache()
    put(BASE/'NATIVE_QUALIFICATION.json',dict(status='pass',count=4,records=rows,clock_proof=proof,new_encoder_calls=6,qualification_feature_caches=2,short_generation_decodes=16,model_load_seconds=secs,GT_read=False))
def teacher(arm):
    import torch,numpy as np
    from scripts.stvg_referent_abc_native002 import processor_batch,pack,sequence_keep,greedy
    from scripts.stvg_native_instance_common_v1 import write
    verify();assert arm in ARMS and read(BASE/'NATIVE_QUALIFICATION.json')['status']=='pass';sys.addaudithook(guard);m,p,device,proof,secs=load();files=[];start=time.perf_counter();vision=fresh=skipped=0;jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records']
    for orig in jobs:
        j=job_for(orig,arm);cache=BASE/'native_cache'/j['job_id'];tick=time.perf_counter();newvision=False
        if not(cache/'POLICIES.json').exists():
            assert arm=='Full_E1';b,ps,e,pos,f,runs,pol,v,ma,gr,q,c,available=prepare(m,p,device,orig);cache_write(orig,proof,ps,f,pol,v,ma,gr,q,c,available);newvision=True;vision+=1
        else:
            s=read(cache/'POLICIES.json');assert s['job_sha256']==hashlib.sha256(json.dumps(orig,sort_keys=True).encode()).hexdigest()and s['model_clock_proof']==proof and sha(cache/'native.npz')==s['native_sha256'];available=s['caption_available'];pol=s['records'];b,ps=processor_batch(p,j,ROOT,device)
            with np.load(cache/'native.npz',allow_pickle=False)as a,torch.inference_mode():e,pos,_,f,runs=pack(m,b,torch.from_numpy(a['features']).to(device=device,dtype=m.dtype))
        if arm in ('B','C')and not available:
            raw='';parsed=dict(status='unavailable_no_caption',selected_candidate_id=None);audit=None;tokens=[];skipped+=1;indices=[]
        else:
            k='A'if arm=='A_ORDER'else arm;indices=[list(range(512)),list(range(768))]if arm in ('Full_E1','FULL_ORDER')else[r['indices'][k]for r in pol];keep=sequence_keep(b,m,runs,indices)
            with torch.inference_mode():tokens,audit=greedy(m,e,pos,keep,b['input_ids'],96)
            raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();parsed=parse(raw,j);fresh+=1
        dest=BASE/'teacher'/arm/(j['job_id']+'.json');put(dest,dict(job_id=j['job_id'],case_id=j['case_id'],dataset=j['dataset'],arm=arm,raw_output=raw,parsed=parsed,job_binding_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),native_policy_sha256=sha(cache/'POLICIES.json'),visual_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),visual_token_count=sum(map(len,indices)),caption_available=available,prompt_sha256=ps,new_vision_call=newvision,new_teacher_call=audit is not None,sequence_readback=audit,output_count=len(tokens),truncated=len(tokens)>=96,wall_seconds=time.perf_counter()-tick,GT_read=False));files.append(dict(case_id=j['case_id'],path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size));write(BASE/'NATIVE_STATUS.json',dict(status='running',stage='Teacher',arm=arm,done=len(files),total=len(jobs),GT_read=False),mutable=True);del b,e,pos,f,runs,pol;gc.collect();torch.cuda.empty_cache()
    put(BASE/'teacher'/arm/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=256,response_count=len(files),files=files,new_vision_calls=vision,new_greedy_generation_calls=fresh,unavailable_no_caption=skipped,model_load_seconds=secs,worker_loop_seconds=time.perf_counter()-start,GT_read=False))
def teacher_root(arm):
    verify();b=BASE/'teacher'/arm;bar=read(b/'TEACHER_GLOBAL_BARRIER.json');jobs={j['job_id']:j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']};assert bar['status']=='complete'and bar['count']==256 and bar['response_count']==len(jobs);rows=[]
    for f in bar['files']:
        assert sha(ROOT/f['path'])==f['sha256'];r=read(ROOT/f['path']);orig=jobs[r['job_id']];j=job_for(orig,arm);c=BASE/'native_cache'/r['job_id'];s=read(c/'POLICIES.json');assert sha(c/'native.npz')==s['native_sha256']and sha(c/'POLICIES.json')==r['native_policy_sha256']and s['job_sha256']==hashlib.sha256(json.dumps(orig,sort_keys=True).encode()).hexdigest()and r['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
        if arm in ('B','C')and not s['caption_available']:assert not r['new_teacher_call']and r['output_count']==0 and r['parsed']['status']=='unavailable_no_caption';ix=[]
        else:
            ix=[list(range(512)),list(range(768))]if arm in ('Full_E1','FULL_ORDER')else[x['indices']['A'if arm=='A_ORDER'else arm]for x in s['records']];assert r['sequence_readback']['all_nonvisual_tokens_preserved']and r['parsed']==parse(r['raw_output'],j)
        assert hashlib.sha256(json.dumps(ix).encode()).hexdigest()==r['visual_indices_sha256'];rows.append(dict(case_id=j['case_id'],dataset=j['dataset'],slot=j['center_slot'],parse_status=r['parsed']['status'],selected_candidate_id=r['parsed'].get('selected_candidate_id'),GT_read=False))
    put(b/'ROOT_READBACK.json',dict(status='pass',count=256,response_count=len(rows),records=rows,barrier_sha256=sha(b/'TEACHER_GLOBAL_BARRIER.json'),GT_read=False))
def supervise(pid):
    fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd);verify()
    from scripts import launch_stvg_instance_identity_v1 as r
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='native256';g=1024**3;v=r.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    try:
        tasks=[('qualification',None)];derived={'A':['A_BT'],'B':['B_BT','B_G10','B_GR11']}
        for a in ARMS:
            tasks += [('teacher',a),('teacher_root',a)]
            for learner in [a]+derived.get(a,[]):tasks += [(s,learner)for s in ('student_prepare','student_worker','student_root')]
        for stage,arm in tasks:
            out=BASE/'execution/native256'/(arm or'qualification')/stage;out.mkdir(parents=True,exist_ok=False);inter='.venv/stvg_native_distill_v1/bin/python'if stage in ('qualification','teacher')else'.conda/tubedetr/bin/python'
            with(out/'worker.log').open('xb')as log:
                t=time.perf_counter();script=ROOT/'scripts/run_stvg_spatial_controls256_student_v1.py'if stage.startswith('student_')else Path(__file__);mode=stage.split('_',1)[1]if stage.startswith('student_')else stage;p=subprocess.Popen([str(ROOT/inter),'-B',str(script),mode]+([arm]if arm else[]),cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0,(stage,arm)
        put(BASE/'TEACHER_HANDOFF.json',dict(status='all_10_Student_measurements_complete_pending_root_score_public',count=256,GT_read=False))
    finally:stop.set()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='after_inputs':supervise(int(sys.argv[2]))
    elif mode=='qualification':qualification()
    elif mode=='teacher':teacher(sys.argv[2])
    else:teacher_root(sys.argv[2])
