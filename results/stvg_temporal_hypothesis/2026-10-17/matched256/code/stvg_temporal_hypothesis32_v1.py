"""Ordered backup B: finite Native joint hypotheses, frozen relative captions,512."""
from pathlib import Path
import gc,hashlib,inspect,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_temporal_semantic_view32_v1 import BASE as A,CORE
BASE=ROOT/'artifacts/stvg_temporal_hypothesis32_v1'
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def lock():
    assert not BASE.exists();BASE.mkdir();write(BASE/'ORDERED_GATE_LOCK.json',dict(status='locked_before_A_scores',count=32,gate='any domain SEM_STUDENT_minus_Native all-source mean vIoU<=0 OR all32 semantic Teacher_to_Native_KL exactly zero',requires_A_CPU_complete=True,cohort_exposed_development=True,not_independent_confirmation=True,no_source_GT_routing=True,no_target_parameter_search=True,GT_read=False))
    for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','BASELINE_BARRIER.json']:(BASE/n).write_bytes((A/n).read_bytes())
def gate():
    assert read(A/'CPU_COMPLETION.json')['count']==32;d=read(A/'POSTSEAL_RESULTS.json');idx={(x['dataset'],x['arm'],x['metric']):x['mean'] for x in d['summary']};deltas={ds:idx[ds,'SEM_STUDENT','m_vIoU']-idx[ds,'Native','m_vIoU'] for ds in ['hc2','vidstg']};u=[u for u in d['usage'] if u['arm']=='SEM'];zero=all(x['Teacher_to_Native_KL']==0 for x in u);trigger=zero or any(v<=0 for v in deltas.values());write(BASE/'DEVELOPMENT_GATE.json',dict(status='triggered' if trigger else 'deferred_A_benefits_both_domains',A_result_sha256=sha(A/'POSTSEAL_RESULTS.json'),per_domain_vIoU_delta=deltas,no_information=zero,GT_read=True,no_source_selection=True));return trigger
def prepare():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case,_render_marked_frames
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f,stvg_factorial_native256_v1 as rgb
    assert read(BASE/'DEVELOPMENT_GATE.json')['status']=='triggered';sys.addaudithook(guard);rgb.BASE=BASE;rows=[];files=[];am={x['case_id']:x for x in read(A/'INPUT_BARRIER.json')['jobs']}
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        c,_=load_case(r);cid=r['probe_alias'];req=read(CORE/'temporal/requests'/(cid+'.json'));available=sorted(x['physical_frame_id'] for x in req['frame_map']);off=f.native.offsets(c,_hidden_from_source_case(c));selection=f.select(off,available);assert selection['Native_interval']==list(c['Native_interval']);hs=selection['hypotheses']
        if len(hs)==2:
            rgb.CURRENT_CASE=c;pixels=rgb.native_rgb(req,set(x for h in hs for x in h['sampled_physical_frame_ids']))
            for k,h in enumerate(hs):
                h['canonical_id']='Native' if k==0 else 'Alternative'
                frames,meta=_render_marked_frames(c,h,h['sampled_physical_frame_ids'],pixels);p=BASE/'inputs'/cid/(str(k)+'.npy');p.parent.mkdir(parents=True,exist_ok=True);np.save(p,frames,allow_pickle=False);h.update(video_path=str(p.relative_to(ROOT)),video_sha256=sha(p),metadata=meta);files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
        if len(hs)==2 and hs[0]['video_sha256']==hs[1]['video_sha256'] and hs[0]['metadata']['clip_fps']==hs[1]['metadata']['clip_fps']:selection['status']='identical_actual_rendered_observations_no_retry'
        aj=read(ROOT/am[cid]['path']);j=dict(source_grid_path=aj['source_grid_path'],source_grid_sha256=aj['source_grid_sha256'],source_fps=aj['source_fps'],legal_bounds=aj['legal_bounds'],case_id=cid,dataset=r['dataset'],query_text=req['query_text'],selection=selection,source_case_sha256=c['source_case_sha256'],GT_read=False);p=BASE/'inputs'/cid/'JOB.json';write(p,j);rows.append(dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p)));print('NATIVE_HYPOTHESES',len(rows),32,flush=True);del c;gc.collect()
    write(BASE/'INPUT_BARRIER.json',dict(status='complete',count=32,jobs=rows,files=files,GT_read=False))
def teacher():
    import numpy as np,torch
    from scripts import run_stvg_referent_abc002 as loader,stvg_table1_temporal_adapter_v1 as ad
    verify();sys.addaudithook(guard);loader.BASE=BASE;m,p,device,proof,secs=loader.load_teacher();outputs=[];qualified=set();counts=dict(captions=0,prefills=0,extra_prefix_qualifications=0)
    def gen(video,fps,cap,claim):
        write(BASE/'claims'/(claim+'.json'),dict(status='claimed_once',GT_read=False));b,ps=ad._processor_batch(p,ad.caption_prompt(),video=video,fps=fps,device=device);n=b['input_ids'].shape[1];t=time.perf_counter()
        with torch.inference_mode():o=m.generate(**b,max_new_tokens=cap,do_sample=False,use_cache=True)
        tok=o[0,n:].cpu().tolist();out=dict(raw=p.tokenizer.decode(tok,skip_special_tokens=True).strip(),tokens=len(tok),truncated=len(tok)>=cap,input_tokens=int(n),prompt_sha256=ps,wall_seconds=time.perf_counter()-t);del b,o;return out,tok
    for jf in read(BASE/'INPUT_BARRIER.json')['jobs']:
        verify();assert sha(ROOT/jf['path'])==jf['sha256'];j=read(ROOT/jf['path']);cid=j['case_id'];hs=j['selection']['hypotheses'];ev=[]
        if len(hs)==2 and j['selection']['status']=='two_distinct_observations':
            for k,h in enumerate(hs):
                assert sha(ROOT/h['video_path'])==h['video_sha256'];video=np.load(ROOT/h['video_path'],allow_pickle=False);cap,tok=gen(video,h['metadata']['clip_fps'],160,cid+'_'+str(k));counts['captions']+=1
                if j['dataset'] not in qualified:
                    small,st=gen(video,h['metadata']['clip_fps'],8,cid+'_qualification8');assert st==tok[:len(st)];qualified.add(j['dataset']);counts['extra_prefix_qualifications']+=1
                write(BASE/'claims'/(cid+'_'+str(k)+'_support.json'),dict(status='claimed_once',GT_read=False));b,ps=ad._processor_batch(p,ad.alignment_prompt(j['query_text'],cap['raw']),device=device);ids=ad._abc_token_ids(p);t=time.perf_counter()
                with torch.inference_mode():o=m(**b,use_cache=False)
                raw=o.logits[0,-1].float().log_softmax(-1)[ids];post=raw.softmax(0).cpu().tolist();reward=post[0]-post[1] if cap['raw'] and not cap['truncated'] else None;ev.append(dict(caption=cap,posterior=post,raw_label_logprobs=raw.cpu().double().tolist(),reward=reward,prompt_sha256=ps,prefill_seconds=time.perf_counter()-t,GT_read=False));counts['prefills']+=1;del b,o,video;torch.cuda.empty_cache()
        signal=len(ev)==2 and all(x['reward'] is not None for x in ev) and ev[0]['reward']!=ev[1]['reward'];dest=BASE/'teacher'/(cid+'.json');write(dest,dict(status='complete',case_id=cid,dataset=j['dataset'],input_sha256=jf['sha256'],evidence=ev,informative=signal,clock_proof=proof,GT_read=False));outputs.append(dict(case_id=cid,path=str(dest.relative_to(ROOT)),sha256=sha(dest)));write(BASE/'TEACHER_STATUS.json',dict(status='running',done=len(outputs),total=32,count=32,GT_read=False),mutable=True)
    write(BASE/'TEACHER_BARRIER.json',dict(status='complete',count=32,files=outputs,calls=counts,qualified_domains=sorted(qualified),load_seconds=secs,GT_read=False))
def teacher_root():
    import numpy as np
    verify();informative=0
    for f in read(BASE/'TEACHER_BARRIER.json')['files']:
        assert sha(ROOT/f['path'])==f['sha256'];d=read(ROOT/f['path'])
        for e in d['evidence']:
            z=np.asarray(e['raw_label_logprobs']);q=np.exp(z-z.max());q/=q.sum();assert np.max(abs(q-e['posterior']))<2e-7
            assert e['reward']==(e['posterior'][0]-e['posterior'][1] if e['caption']['raw'] and not e['caption']['truncated'] else None)
        informative+=d['informative']
    write(BASE/'TEACHER_ROOT_READBACK.json',dict(status='pass',count=32,informative_count=informative,GT_read=False));write(BASE/'SIGNAL_COMPLETION.json',dict(status='complete',count=32,informative_count=informative,Student_fit_only_when_signal=True,GT_read=False))
def fit():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f
    verify();assert read(BASE/'SIGNAL_COMPLETION.json')['status']=='complete';jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};files=[];targets=[];checks=0
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        c,_=load_case(r);cid=r['probe_alias'];j=read(ROOT/jm[cid]['path']);off=f.native.offsets(c,_hidden_from_source_case(c));d=read(BASE/'teacher'/(cid+'.json'));hs=j['selection']['hypotheses'];rew=[e['reward'] for e in d['evidence']];ff=f.fit(off,hs,rew);checks+=f.audit(off,ff);v=list(c['Native_interval']) if ff['no_information'] else hs[int(np.argmax(ff['Teacher_conditional']))]['interval_frames'];tp=save_pt(BASE/'targets'/(cid+'.pt'),dict(view_intervals=dict(HYP=v),GT_read=False));tp['case_id']=cid;targets.append(tp);p=save_pt(BASE/'fits'/(cid+'.pt'),dict(case_id=cid,dataset=r['dataset'],fits=dict(HYP=ff),source_case_sha256=c['source_case_sha256'],GT_read=False));p['case_id']=cid;files.append(p)
    write(BASE/'TARGET_BARRIER.json',dict(status='complete',count=32,files=targets,GT_read=False));write(BASE/'FIT_BARRIER.json',dict(status='complete',count=32,files=files,independent_checks=checks,GT_read=False))
def readout():
    from scripts import stvg_temporal_semantic_grounder32_v1 as ori
    from scripts.stvg_temporal_semantic_grounder32_v2 import clean
    verify();clean();src=inspect.getsource(ori.readout).replace("['SEM','DIRECT','PARA']","['HYP']");ns=dict(ori.__dict__);ns.update(BASE=BASE,verify=verify);exec(compile(src,__file__,'exec'),ns);ns['readout']();write(BASE/'PRIMARY_COMPLETION.json',dict(status='complete',count=32,GT_read=False))
if __name__=='__main__':globals()[sys.argv[1]]()
