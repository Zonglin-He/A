"""Fixed B3 hypotheses and Student, only frozen Teacher family changes."""
from pathlib import Path
import gc,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_qwen_b3_temporal32_v1';H=ROOT/'artifacts/stvg_native_head_adapter32_v1';B2=ROOT/'artifacts/stvg_temporal_marginal_train256_v1';B3=ROOT/'artifacts/stvg_temporal_marginal_decode256_v1';Q=ROOT/'artifacts/stvg_qwen35_geometry32_v1'
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def init():
    assert not BASE.exists();BASE.mkdir();(BASE/'ROSTER_LOCK.json').write_bytes((H/'ROSTER_LOCK.json').read_bytes());ids={r['probe_alias']for r in read(BASE/'ROSTER_LOCK.json')['records']}
    jobs=[f for f in read(B2/'INPUT_BARRIER.json')['jobs']if f['case_id']in ids];assert len(jobs)==32
    write(BASE/'INPUT_BARRIER.json',dict(status='complete',count=32,jobs=jobs,exact_B3_inputs_reused=True,GT_read=False))
    names=['stvg_qwen_b3_temporal32_v1.py','continue_stvg_qwen_b3_temporal32_v1.py','report_stvg_qwen_b3_temporal32_v1.py'];paths=[ROOT/'scripts'/n for n in names]+[ROOT/'protocols/stvg_qwen_b3_temporal32_v1.md',ROOT/'vg_tta/stvg_native_head_residual32_v1.py',ROOT/'scripts/stvg_temporal_marginal_train_fit256_v1.py',ROOT/'scripts/stvg_temporal_marginal_decode_util256_v1.py',BASE/'ROSTER_LOCK.json',BASE/'INPUT_BARRIER.json',B2/'TEACHER_BARRIER.json',B3/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json',Q/'MODEL_SNAPSHOT_LOCK.json',Q/'EXECUTION_RUNTIME003.json',ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json']
    for j in jobs:
        assert sha(ROOT/j['path'])==j['sha256'];paths.append(ROOT/j['path'])
        for h in read(ROOT/j['path'])['selection']['hypotheses']:assert sha(ROOT/h['video_path'])==h['video_sha256'];paths.append(ROOT/h['video_path'])
    write(BASE/'METHOD_LOCK.json',dict(status='locked_before_model',count=32,per_domain=16,new_captions_max=64,new_alignment_prefills_max=64,extra_prefix_qualification_generations_max=2,only_Teacher_changed=True,Student_parameters=512,B3_hypotheses_pixels_rewards_task_optimizer_marginal_decoder_fixed=True,Actionness=False,GT_read=False));paths.append(BASE/'METHOD_LOCK.json');write(BASE/'EXECUTION_RUNTIME.json',dict(status='locked',count=32,pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False));write(BASE/'STATUS.json',dict(status='prepared',count=32,GT_read=False),mutable=True)

def teacher():
    import numpy as np,torch
    from transformers.video_utils import VideoMetadata
    from scripts.stvg_qwen35_geometry_adapters_v2 import load_future_model
    from scripts.stvg_table1_temporal_adapter_v1 import caption_prompt,alignment_prompt,_abc_token_ids,configure_original_processor
    verify();sys.addaudithook(guard);assert read(Q/'CPU_COMPLETION.json')['status']=='complete';model,p=load_future_model();model.requires_grad_(False);configure_original_processor(p);outputs=[];qualified=set();counts=dict(captions=0,prefills=0,extra_prefix_qualifications=0)
    def batch(text,video=None,fps=None):
        messages=[dict(role='user',content=([dict(type='video')]if video is not None else [])+[dict(type='text',text=text)])]
        rendered=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);assert rendered.endswith('<think>\n\n</think>\n\n')
        kw=dict(text=[rendered],return_tensors='pt',add_special_tokens=False)
        if video is not None:kw.update(videos=[torch.from_numpy(video.copy())],do_sample_frames=False,video_metadata=[VideoMetadata(total_num_frames=8,fps=float(fps),frames_indices=list(range(8)),duration=8/float(fps))],return_metadata=True)
        b=p(**kw);b.pop('video_metadata',None);assert b['input_ids'].shape[1]<=4096
        if video is not None:
            grid=b['video_grid_thw'][0];assert int(grid[0])==4;assert int((b['input_ids']==248057).sum())==int(grid.prod())//4
            text2=p.tokenizer.decode(b['input_ids'][0].tolist(),skip_special_tokens=False)
            for k in [0,2,4,6]:assert f'<{(k+.5)/float(fps):.1f} seconds>'in text2
        return b.to(model.device),hashlib.sha256(rendered.encode()).hexdigest()
    def gen(video,fps,cap,claim):
        write(BASE/'claims'/(claim+'.json'),dict(status='claimed_once',GT_read=False));b,ps=batch(caption_prompt(),video,fps);n=b['input_ids'].shape[1];tick=time.perf_counter()
        with torch.inference_mode():out=model.generate(**b,max_new_tokens=cap,do_sample=False,use_cache=True)
        torch.cuda.synchronize();tok=out[0,n:].cpu().tolist();d=dict(raw=p.tokenizer.decode(tok,skip_special_tokens=True).strip(),tokens=len(tok),truncated=len(tok)>=cap,input_tokens=int(n),prompt_sha256=ps,wall_seconds=time.perf_counter()-tick);del b,out;return d,tok
    for jf in read(BASE/'INPUT_BARRIER.json')['jobs']:
        verify();assert sha(ROOT/jf['path'])==jf['sha256'];j=read(ROOT/jf['path']);cid=j['case_id'];ev=[]
        if len(j['selection']['hypotheses'])==2 and j['selection']['status']=='two_distinct_observations':
            for k,h in enumerate(j['selection']['hypotheses']):
                assert sha(ROOT/h['video_path'])==h['video_sha256'];video=np.load(ROOT/h['video_path'],allow_pickle=False);assert video.shape==(8,3,448,448);fps=h['metadata']['clip_fps'];cap,tok=gen(video,fps,160,cid+'_'+str(k));counts['captions']+=1
                if j['dataset']not in qualified:
                    small,st=gen(video,fps,8,cid+'_qualification8');assert st==tok[:len(st)];qualified.add(j['dataset']);counts['extra_prefix_qualifications']+=1
                write(BASE/'claims'/(cid+'_'+str(k)+'_support.json'),dict(status='claimed_once',GT_read=False));b,ps=batch(alignment_prompt(j['query_text'],cap['raw']));ids=_abc_token_ids(p);tick=time.perf_counter()
                with torch.inference_mode():out=model(**b,use_cache=False)
                torch.cuda.synchronize();raw=out.logits[0,-1].float().log_softmax(-1)[ids];post=raw.softmax(0).cpu().tolist();reward=post[0]-post[1]if cap['raw']and not cap['truncated']else None;ev.append(dict(caption=cap,posterior=post,raw_label_logprobs=raw.cpu().double().tolist(),reward=reward,prompt_sha256=ps,prefill_seconds=time.perf_counter()-tick,GT_read=False));counts['prefills']+=1;del b,out,video;torch.cuda.empty_cache()
        informative=len(ev)==2 and all(e['reward']is not None for e in ev)and ev[0]['reward']!=ev[1]['reward'];dest=BASE/'teacher'/(cid+'.json');write(dest,dict(status='complete',case_id=cid,dataset=j['dataset'],input_sha256=jf['sha256'],evidence=ev,informative=informative,GT_read=False));outputs.append(dict(case_id=cid,path=str(dest.relative_to(ROOT)),sha256=sha(dest)));write(BASE/'STATUS.json',dict(status='running',stage='Teacher',done=len(outputs),total=32,count=32,GT_read=False),mutable=True);print('QWEN_B3_TEACHER',len(outputs),flush=True)
    assert qualified=={'hc2','vidstg'};write(BASE/'TEACHER_BARRIER.json',dict(status='complete',count=32,files=outputs,calls=counts,qualified_domains=sorted(qualified),GT_read=False))
    write(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,actual_eight_frame_clock_tokens_and_greedy_prefix_exact=True,same_B3_candidate_bytes=True,GT_read=False))

def fit():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_marginal_train_fit256_v1 as f
    verify();sys.addaudithook(guard);assert read(BASE/'TEACHER_BARRIER.json')['count']==32;jm={j['case_id']:j for j in read(BASE/'INPUT_BARRIER.json')['jobs']};files=[];checks=0;informative=0
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        c,_=load_case(r);cid=r['probe_alias'];j=read(ROOT/jm[cid]['path']);d=read(BASE/'teacher'/(cid+'.json'))
        for e in d['evidence']:
            z=np.array(e['raw_label_logprobs']);q=np.exp(z-z.max());q/=q.sum();assert np.max(abs(q-e['posterior']))<2e-7;assert e['reward']==(e['posterior'][0]-e['posterior'][1]if e['caption']['raw']and not e['caption']['truncated']else None)
        off=f.native.offsets(c,_hidden_from_source_case(c));ff=f.fit(off,j['selection']['hypotheses'],[e['reward']for e in d['evidence']]);checks+=f.audit(off,ff);informative+=not ff['no_information'];v=save_pt(BASE/'fits'/(cid+'.pt'),dict(case_id=cid,dataset=r['dataset'],fit=ff,source_case_sha256=c['source_case_sha256'],GT_read=False));v['case_id']=cid;files.append(v);del c,off,ff;gc.collect()
    write(BASE/'FIT_BARRIER.json',dict(status='complete',count=32,files=files,independent_path_gradient_checks=checks,GT_read=False));write(BASE/'TEACHER_ROOT_READBACK.json',dict(status='pass',count=32,informative_count=informative,GT_read=False))

def readout():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.stvg_temporal_semantic_grounder32_v1 import cached_source
    from scripts.stvg_temporal_semantic_grounder32_v2 import clean
    from scripts.run_decota_paper_main_v1 import model_for
    from vg_tta.stvg_native_head_residual32_v1 import TemporalHeadResidual
    from scripts.stvg_temporal_marginal_decode_util256_v1 import decode
    verify();clean();sys.addaudithook(guard);files=[];model=None;source=None;fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};bm={f['case_id']:f for f in read(B3/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};jm={j['case_id']:j for j in read(BASE/'INPUT_BARRIER.json')['jobs']}
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        verify();cid=r['probe_alias'];c,_=load_case(r);src='vidstg'if r['dataset']=='hc2'else'hcstvg2'
        if src!=source:
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            model=model_for(src);source=src
        f=fm[cid];assert sha(ROOT/f['path'])==f['sha256'];fit=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)['fit'];base=cached_source(model,c);head=base.head;base.head=TemporalHeadResidual(head,fit['theta'])
        with torch.no_grad():v=base.reference_values(capture=False)
        assert torch.equal(v['boxes'].cpu(),c['zero']['boxes']) and all(torch.equal(a.cpu(),b)for a,b in zip(v['logits'],fit['final_logits']));iv=decode(c['records'],v['logits'])[0];old=bm[cid];assert sha(ROOT/old['path'])==old['sha256'];prior=torch.load(ROOT/old['path'],map_location='cpu',weights_only=False);j=read(ROOT/jm[cid]['path']);native=np.asarray(c['native_boxes_xyxy_normalized']).copy();miv=prior['arms']['MARGINAL_NO_UPDATE']['interval'];hard=miv if fit['no_information']else j['selection']['hypotheses'][int(np.argmax(fit['Teacher_conditional']))]['interval_frames']
        arms=dict(Frozen=dict(boxes=native,interval=list(c['Native_interval'])),MARGINAL_NO_UPDATE=dict(boxes=native,interval=miv),R1_B3=prior['arms']['B_MARGINAL_DECODE'],QWEN_B3=dict(boxes=native,interval=iv),QWEN_TARGET_HARD=dict(boxes=native,interval=hard));f=save_pt(BASE/'predictions'/(cid+'.pt'),dict(case_id=cid,dataset=r['dataset'],arms=arms,actual_integrated_head_logits=v['logits'],GT_read=False));f['case_id']=cid;files.append(f);del base,c,v,fit;gc.collect();torch.cuda.empty_cache()
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));write(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,all_actual_head_outputs_exact=True,spatial_Source_unchanged=True,GT_read=False))

def score():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass';bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    write(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,GT_read=False));rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();rows=[];usage=[];r1={f['case_id']:f for f in read(B2/'TEACHER_BARRIER.json')['files']};jm={j['case_id']:j for j in read(BASE/'INPUT_BARRIER.json')['jobs']};fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']}
    for r,f in zip(rr,bar['files']):
        cid=r['probe_alias'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);fit=torch.load(ROOT/fm[cid]['path'],map_location='cpu',weights_only=False)['fit'];new=read(BASE/'teacher'/(cid+'.json'));old=read(ROOT/r1[cid]['path']);j=read(ROOT/jm[cid]['path']);native=d['arms']['Frozen'];hyp=[]
        for a,p in d['arms'].items():
            m,_=metric._score_student_arm(p,cases[cid],targets[cid],set(),off,ind);m.update(R_vIoU_at_0p3=float(m['m_vIoU']>=.3),R_vIoU_at_0p5=float(m['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=r['dataset'],arm=a,metrics=m))
        for h in j['selection']['hypotheses']:
            m,_=metric._score_student_arm(dict(native,interval=h['interval_frames']),cases[cid],targets[cid],set(),off,ind);hyp.append(dict(tIoU=m['m_tIoU'],vIoU=m['m_vIoU']))
        usage.append(dict(case_id=cid,dataset=r['dataset'],R1_rewards=[e['reward']for e in old['evidence']],Qwen_rewards=[e['reward']for e in new['evidence']],R1_posteriors=[e['posterior']for e in old['evidence']],Qwen_posteriors=[e['posterior']for e in new['evidence']],Qwen_allowed_label_mass=[float(np.exp(e['raw_label_logprobs']).sum())for e in new['evidence']],R1_informative=old['informative'],Qwen_informative=new['informative'],hypothesis_metrics=hyp,optimizer_steps=fit['optimizer_steps'],termination=fit['termination'],initial_candidate_KL=fit['path'][0]['terms']['candidate_kl'],final_candidate_KL=fit['path'][-1]['terms']['candidate_kl'],Native_conditional=list(fit['Native_conditional']),Teacher_conditional=list(fit['Teacher_conditional'])))
    summary,paired=subset_aggregate(rows);write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,usage=usage,GT_bindings=bindings,GT_read=True,development_not_confirmation=True,only_Teacher_changed=True));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True))
if __name__=='__main__':globals()[sys.argv[1]]()
