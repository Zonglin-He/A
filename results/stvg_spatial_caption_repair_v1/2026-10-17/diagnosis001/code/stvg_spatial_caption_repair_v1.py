"""Spatial caption eligibility is per clip, independent of four-way Temporal fit.

The original Temporal adapter and its four-distinct-candidate gate are unchanged.
Historical locks, predictions and captions are immutable. New artifacts are
versioned, GT-free repairs; raw text/media remain private.
"""
from pathlib import Path
import collections,copy,gc,hashlib,inspect,json,os,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write
from scripts import stvg_factorial_native256_v1 as original
BASE=ROOT/'artifacts/stvg_spatial_caption_repair_v1'
NATIVE=original.BASE;CORE=original.CORE
FIXED=ROOT/'artifacts/stvg_factorial_native256_captionfix002'
CONF=ROOT/'artifacts/stvg_overnight_consolidation_20261016/confirmation128'
def put(p,d,replace=False):return write(p,d,mutable=replace)

def candidate_eligibility(case,request,candidate):
    """Validate exactly one existing candidate; never invent/pad a fourth."""
    from scripts import stvg_table1_temporal_adapter_v1 as a
    from scripts.run_stvg_table1_native_instance_adapter_v1 import _native
    try:
        cid=a._candidate_id(candidate);iv=candidate['interval_frames']
        if len(iv)!=2 or not all(__import__('math').isfinite(float(x)) for x in iv) or iv[0]>=iv[1]:return False,'invalid_interval',None
        req=_native()._normalize_request(request);mapping=a._frame_map_rows(req['frame_map']);available=sorted(mapping)
        if any(b!=a+1 for a,b in zip(available,available[1:])):return False,'noncontiguous_request_clock',None
        roles=[r for r in case['candidate_validity']['candidate_roles'] if r.get('candidate_id')==cid]
        if len(roles)!=1 or roles[0].get('clip_request_eligible') is not True:return False,'individual_candidate_not_eligible',None
        sampled,stride=a._sample_candidate_ids(candidate,available)
        if list(map(int,roles[0].get('clip',{}).get('sampled_physical_frame_ids',[])))!=sampled:return False,'individual_sample_binding_mismatch',None
        return True,None,dict(candidate_id=cid,sampled_physical_frame_ids=sampled,uniform_stride=stride)
    except (KeyError,ValueError,TypeError,IndexError) as e:return False,'individual_clock_invalid_'+type(e).__name__,None

def verify():
    original.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME001.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (FIXED/'EXECUTION_RUNTIME001.json').exists():
        for p,h in read(FIXED/'EXECUTION_RUNTIME001.json')['pins'].items():assert sha(ROOT/p)==h,p

def setup():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    BASE.mkdir(exist_ok=False);records=[];summary=[];same=[]
    for label,b in [('core256',CORE),('confirmation128',CONF)]:
        caps=read(b/'CAPTION_LOCK.json')['records'];rr={r['probe_alias']:r for r in read(b/'ROSTER_LOCK.json')['records']};inp={r['case_id']:r for r in read(b/'temporal/INPUT_BARRIER.json')['records']};reasons=collections.Counter();domains=collections.Counter();existing=0
        for cap in caps:
            cid=cap['case_id'];ip=inp[cid]
            if cap['available']:
                assert sha(ROOT/cap['path'])==cap['sha256'] and read(ROOT/cap['path'])['raw_caption'].strip();existing+=1;continue
            case,_=load_case(rr[cid]);req=read(b/'temporal/requests'/(cid+'.json'));candidate=next(x for x in case['temporal_candidates'] if x['canonical_id']=='time_Native');ok,reason,binding=candidate_eligibility(case,req,candidate);assert ok,(label,cid,reason)
            assert ip['eligible'] is False and not (b/'temporal/teacher'/cid/'time_Native.json').exists();assert read(b/'TEMPORAL_LOCK.json')['records'][list(rr).index(cid)]['selected_interval']==list(case['Native_interval'])
            reasons[ip['fallback_reason']]+=1;domains[rr[cid]['dataset']]+=1
            records.append(dict(cohort=label,case_id=cid,record=rr[cid],request_path=str((b/'temporal/requests'/(cid+'.json')).relative_to(ROOT)),request_sha256=sha(b/'temporal/requests'/(cid+'.json')),candidate=candidate,binding=binding,whole_Temporal_skip_reason=ip['fallback_reason'],caption_never_requested=True,Source_Native_equals_fallback_T=True,GT_read=False));del case;gc.collect()
        summary.append(dict(cohort=label,count=len(caps),existing_captions=existing,missing=len(caps)-existing,missing_by_domain=dict(domains),reasons=dict(reasons),empty_model_outputs=0,missing_after_generation=0,GT_read=False))
    assert len(records)==54
    put(BASE/'DIAGNOSIS.json',dict(status='complete',count=384,records=records,summary=summary,root_cause='whole four-candidate Temporal admission was incorrectly reused as the prerequisite for independent Spatial Caption acquisition',Temporal_fallback_rule_preserved=True,new_GT_reader_calls=0,new_model_calls=0,GT_read=False))
    # Preserve only disposable old R1 features needed to match same-clock
    # controls. Hard links survive the already-authorized cache retirement.
    target=BASE/'preserved_matched_features';target.mkdir();oldtimes={r['case_id']:r['selected_interval'] for r in read(CORE/'TEMPORAL_LOCK.json')['records']};nt={r['case_id']:r['selected_interval'] for r in read(NATIVE/'TEMPORAL_LOCK.json')['records']};oldjobs=read(CORE/'TEACHER_JOB_LOCK.json')['records'];saved=[]
    for j in oldjobs:
        if oldtimes[j['case_id']]!=nt[j['case_id']]:continue
        same.append(j['case_id']);p=CORE/'native_cache'/j['job_id']/'native.npz';policy=read(p.with_name('POLICIES.json'));dest=target/(j['job_id']+'.npz')
        if p.exists():
            try:os.link(p,dest)
            except FileNotFoundError:pass
        saved.append(dict(job_id=j['job_id'],native_sha256=policy['native_sha256'],preserved_path=str(dest.relative_to(ROOT)) if dest.exists() else None,GT_read=False))
    put(BASE/'SAME_CLOCK_CACHE_INVENTORY.json',dict(status='locked',count=len(set(same)),records=saved,reason='same Native/Adapted interval uses the exact existing matched E1 pool and pixels rather than rerendering the same control',GT_read=False))
    put(BASE/'REPAIR_PLAN_LOCK.json',dict(status='locked',count=54,new_plain_caption_calls=54,maximum_extra_qualification_caption_calls=2,maximum_extra_qualification_alignment_calls=2,caption_frames=8,side=448,caption_token_cap=160,greedy=True,no_new_Temporal_parameter_updates=True,no_Temporal_four_candidate_gate_change=True,old_results_preserved=True,old_confirmation_not_relabelled=True,fixed_Native_count=256,missing_core_caption_count=35,Native_clocks_unchanged=True,same_clock_pools_exactly_reused=True,paired_Full_B_caption_repair_only_reuses_exact_matching_Native_B_outputs_on_35_fallback_sources=True,no_new_GT_for_caption_repair=True,GT_read=False))
    put(BASE/'STATUS.json',dict(status='prepared',stage='missing54_single_candidate_CPU_inputs',count=54,GT_read=False));print(json.dumps(dict(status='prepared',missing54=54,summary=summary,same_clock_sources=len(set(same)),preserved_features=sum(x['preserved_path'] is not None for x in saved))))

def prepare():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import stvg_table1_temporal_adapter_v1 as a
    verify();files=[]
    for r in read(BASE/'DIAGNOSIS.json')['records']:
        case,_=load_case(r['record']);req=read(ROOT/r['request_path']);ok,why,binding=candidate_eligibility(case,req,r['candidate']);assert ok and binding==r['binding'];ns=dict(original.__dict__);ns.update(BASE=BASE,CURRENT_CASE=case);exec(compile(inspect.getsource(original.native_rgb),__file__,'exec'),ns);rgb=ns['native_rgb'](req,binding['sampled_physical_frame_ids']);video,meta=a._render_marked_frames(case,r['candidate'],binding['sampled_physical_frame_ids'],rgb);assert video.shape==(8,3,448,448) and video.dtype==np.uint8
        dest=BASE/'private_inputs'/r['cohort']/(r['case_id']+'.npy');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.save(dest,video,allow_pickle=False);mp=dest.with_suffix('.json');put(mp,dict(**r,metadata=meta,input_path=str(dest.relative_to(ROOT)),input_sha256=sha(dest),clip_fps=meta['clip_fps']));files.append(dict(cohort=r['cohort'],case_id=r['case_id'],path=str(mp.relative_to(ROOT)),sha256=sha(mp),GT_read=False));put(BASE/'INPUT_STATUS.json',dict(status='running',done=len(files),total=54,count=54,GT_read=False),replace=True);del case,rgb,video;gc.collect()
    put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=54,files=files,all_individual_8frame_Source_RGB_and_canonical_pixel_bindings_exact=True,Temporal_gate_unchanged=True,GT_read=False));print('CAPTION54_SINGLE_CLIP_INPUTS_COMPLETE')

def generate():
    import torch,numpy as np
    from scripts import run_stvg_native_instance_teacher_storage002 as loader,stvg_table1_temporal_adapter_v1 as a
    from scripts.stvg_qwen_uniform_clock002 import apply
    from scripts.stvg_native_instance_common_v1 import guard
    verify();sys.addaudithook(guard);torch.set_num_threads(2);loader.verify_local_model_receipt();loader.MAX_MEMORY={0:'20GiB','cpu':'4GiB'};_,m,_,device,_,load_seconds,_=loader._load_qwen(ROOT/loader.CHECKPOINT_REL,BASE/'offload');m.generation_config.use_cache=True;proof=apply(m);a.configure_original_processor(p:=__import__('transformers').AutoProcessor.from_pretrained(str(ROOT/loader.CHECKPOINT_REL),local_files_only=True,trust_remote_code=False));files=[];qualified=set();qcalls=0
    for f in read(BASE/'INPUT_BARRIER.json')['files']:
        meta=read(ROOT/f['path']);assert sha(ROOT/f['path'])==f['sha256'] and sha(ROOT/meta['input_path'])==meta['input_sha256'];video=np.load(ROOT/meta['input_path'],allow_pickle=False);b,ps=a._processor_batch(p,a.caption_prompt(),video=video,fps=meta['clip_fps'],device=device);n=b['input_ids'].shape[1]
        with torch.inference_mode():out=m.generate(**b,max_new_tokens=a.CAPTION_TOKEN_CAP,do_sample=False,use_cache=True)
        text=p.tokenizer.decode(out[0,n:],skip_special_tokens=True).strip();tokens=out[0,n:].cpu().tolist();assert text,'empty caption remains an explicit failure, never fabricated'
        ds=meta['record']['dataset']
        if ds not in qualified:
            query=read(ROOT/meta['request_path'])['query_text'];comparison=a.run_hot_candidate_teacher(m,p,query,meta['candidate'],dict(video=video,metadata=meta['metadata'],fps=meta['clip_fps']),device=device);assert comparison['raw_caption']==text and comparison['caption_prompt_sha256']==ps and comparison['caption_output_tokens']==len(tokens);qualified.add(ds);qcalls+=1
        dest=BASE/'private_captions'/meta['cohort']/(meta['case_id']+'.json');put(dest,dict(status='actual_plain_description',candidate_id='time_Native',raw_caption=text,caption_sha256=hashlib.sha256(text.encode()).hexdigest(),caption_prompt_sha256=ps,caption_output_tokens=len(tokens),output_truncated_at_cap=len(tokens)>=160,metadata=meta['metadata'],input_sha256=meta['input_sha256'],corrected_clock_proof=proof,new_plain_caption_call=True,alignment_or_Temporal_fit_not_required_for_spatial_caption=True,teacher_weight_update=False,GT_read=False));files.append(dict(cohort=meta['cohort'],case_id=meta['case_id'],path=str(dest.relative_to(ROOT)),sha256=sha(dest),GT_read=False));put(BASE/'CAPTION_STATUS.json',dict(status='running',done=len(files),total=54,count=54,GT_read=False),replace=True);del b,out,video;gc.collect();torch.cuda.empty_cache()
    assert qualified=={'hc2','vidstg'};put(BASE/'CAPTION_COMPLETION.json',dict(status='complete',count=54,files=files,new_plain_caption_calls=54,engineering_extra_caption_calls=qcalls,engineering_alignment_calls=qcalls,exact_original_caption_qualification_domains=sorted(qualified),new_Temporal_fits=0,model_load_seconds=load_seconds,GT_read=False));print('CAPTION54_ACTUAL_NONEMPTY_AND_ORIGINAL_GENERATOR_QUALIFIED')

def setup_fixed():
    verify();assert read(NATIVE/'FEATURE_BARRIER.json')['count']==256;assert read(BASE/'CAPTION_COMPLETION.json')['count']==54
    FIXED.mkdir(exist_ok=False);rr=read(NATIVE/'ROSTER_LOCK.json');put(FIXED/'ROSTER_LOCK.json',rr)
    for name in ('SOURCE_CAPTURE_BARRIER.json','SOURCE_GEOMETRY.json','TEMPORAL_LOCK.json','SPATIAL_INPUT_HANDOFF.json','EVIDENCE_MODIFICATION_LOCK.json','ROOT_NATIVE_CLOCK_SOURCE_READBACK.json'):(FIXED/name).write_bytes((NATIVE/name).read_bytes())
    (FIXED/'temporal').mkdir();(FIXED/'temporal/requests').symlink_to(NATIVE/'temporal/requests',target_is_directory=True)
    repaired={r['case_id']:r for r in read(BASE/'CAPTION_COMPLETION.json')['files'] if r['cohort']=='core256'};caps=[]
    for r in read(NATIVE/'CAPTION_LOCK.json')['records']:
        if r['available']:caps.append(r)
        else:
            f=repaired[r['case_id']];assert read(ROOT/f['path'])['raw_caption'].strip();caps.append(dict(case_id=r['case_id'],candidate_id='time_Native',path=f['path'],sha256=f['sha256'],available=True,repaired_from_individual_Native_clip=True,GT_read=False))
    assert len(caps)==256 and all(r['available'] for r in caps);put(FIXED/'CAPTION_LOCK.json',dict(status='locked',count=256,records=caps,individual_caption_acquisition_independent_of_Temporal_admission=True,GT_read=False))
    same={r['job_id'] for r in read(BASE/'SAME_CLOCK_CACHE_INVENTORY.json')['records']};old={(j['case_id'],j['center_slot']):j for j in read(CORE/'TEACHER_JOB_LOCK.json')['records']};jobs=[];replaced=set()
    for j in read(NATIVE/'TEACHER_JOB_LOCK.json')['records']:
        if j['job_id'] in same:j=old[j['case_id'],j['center_slot']];replaced.add(j['case_id'])
        jobs.append(j)
    put(FIXED/'TEACHER_JOB_LOCK.json',dict(status='locked',count=256,job_count=len(jobs),records=jobs,GT_read=False));prop=FIXED/'spatial/E1/proposals';prop.mkdir(parents=True)
    for r in rr['records']:
        cid=r['probe_alias'];p=(CORE if cid in replaced else NATIVE)/'spatial/E1/proposals'/(cid+'.json');(prop/(cid+'.json')).symlink_to(p)
    study=copy.deepcopy(read(NATIVE/'STUDY_LOCK.json'));study.update(caption_input_repair=True,original_scientific_plan_sha256=sha(NATIVE/'STUDY_LOCK.json'),original_caption_missing_branch_results_retained=True,same_clock_exact_E1_control_reuse_count=len(replaced),no_new_scientific_parameters_or_Temporal_updates=True);put(FIXED/'STUDY_LOCK.json',study)
    for arm in ('A','B'):
        b=FIXED/'arms'/arm;b.mkdir(parents=True);put(b/'ROSTER_LOCK.json',rr);cfg=read(NATIVE/'arms'/arm/'SCOPE_CONFIG.json');cfg.update(core_scope=str(FIXED.relative_to(ROOT)),label='Native-clock '+arm+' with individually repaired Caption acquisition');put(b/'SCOPE_CONFIG.json',cfg)
    protected=[BASE/'EXECUTION_RUNTIME001.json',FIXED/'CAPTION_LOCK.json',FIXED/'TEACHER_JOB_LOCK.json',FIXED/'STUDY_LOCK.json',FIXED/'ROSTER_LOCK.json',FIXED/'SOURCE_CAPTURE_BARRIER.json',FIXED/'TEMPORAL_LOCK.json']+[FIXED/'arms'/a/'SCOPE_CONFIG.json' for a in ('A','B')]
    put(FIXED/'EXECUTION_RUNTIME001.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in protected},deadline_epoch=read(NATIVE/'EXECUTION_RUNTIME001.json')['deadline_epoch'],original_finite_window_not_extended=True,GT_read=False))
    put(FIXED/'ROOT_INPUT_READBACK.json',dict(status='pass',count=256,job_count=len(jobs),same_clock_controls_use_exact_existing_E1_inputs=True,caption_count=256,geometry_and_Source_Native_intervals_unchanged=True,GT_read=False));put(FIXED/'STATUS.json',dict(status='prepared',stage='features',count=256,GT_read=False))

def features_fixed():
    import torch,numpy as np
    from scripts import run_stvg_spatial_controls256_native_v1 as g
    verify();g.BASE=FIXED;m,p,device,proof,seconds=g.load();inventory={r['job_id']:r for r in read(BASE/'SAME_CLOCK_CACHE_INVENTORY.json')['records']};files=[];new=0;reused=0
    for j in read(FIXED/'TEACHER_JOB_LOCK.json')['records']:
        dest=FIXED/'native_cache'/j['job_id'];dest.parent.mkdir(exist_ok=True)
        if j['job_id'] not in inventory:
            prior=NATIVE/'native_cache'/j['job_id'];policy=read(prior/'POLICIES.json');assert policy['caption_available'] and sha(prior/'native.npz')==policy['native_sha256'];dest.symlink_to(prior,target_is_directory=True);reused+=1
        else:
            item=inventory[j['job_id']];frozen=None
            if item['preserved_path']:
                path=ROOT/item['preserved_path'];assert sha(path)==item['native_sha256']
                with np.load(path,allow_pickle=False) as z:frozen=z['features'].copy()
            b,ps,e,pos,f,runs,pol,v,ma,gr,q,c,available=g.prepare(m,p,device,j,frozen=frozen);assert available;g.cache_write(j,proof,ps,f,pol,v,ma,gr,q,c,available);prior=read(CORE/'native_cache'/j['job_id']/'POLICIES.json')
            for k in (0,1):assert pol[k]['indices']['A'].tolist()==prior['records'][k]['indices']['A']
            new+=frozen is None;reused+=frozen is not None;del b,e,pos,f,runs,pol,frozen;gc.collect();torch.cuda.empty_cache()
        path=dest/'POLICIES.json';files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),case_id=j['case_id']));put(FIXED/'FEATURE_STATUS.json',dict(status='running',done=len(files),total=1024,count=256,GT_read=False),replace=True)
    put(FIXED/'FEATURE_BARRIER.json',dict(status='complete',count=256,files=files,new_encoder_calls=new,exact_visual_reuses=reused,all_Captions_available=True,same_clock_A_policies_exact=True,model_load_seconds=seconds,GT_read=False));print('REPAIRED_NATIVE_FEATURES_COMPLETE')

def configure(module,arm):
    verify();scope=FIXED/'arms'/arm;module.BASE=scope;module.CORE=FIXED;module.N=256;module.CFG=read(scope/'SCOPE_CONFIG.json');module.common=types.SimpleNamespace(BASE=FIXED,CORE=CORE,verify=verify,deadline=lambda:read(NATIVE/'EXECUTION_RUNTIME001.json')['deadline_epoch'],read=read,write=put,sha=sha);return module

def arm_stage(arm,stage):
    if stage in ('jobs','qualification','teacher','teacher_root'):
        from scripts import stvg_overnight_global_v1 as g
        configure(g,arm);return {'jobs':g.jobs,'qualification':g.qualify,'teacher':g.teacher,'teacher_root':g.teacher_root}[stage]()
    from scripts import stvg_overnight_student_v1 as s
    configure(s,arm)
    if stage=='student_prepare':
        src=inspect.getsource(s.prepare);needle="read(CORE/'TEMPORAL_LOCK.json')";assert src.count(needle)==1;src=src.replace(needle,"read(ORIGINAL_CORE/'TEMPORAL_LOCK.json')");ns=dict(s.__dict__);ns['ORIGINAL_CORE']=CORE;exec(compile(src,__file__,'exec'),ns);return ns['prepare']()
    return {'student_worker':s.worker,'student_root':s.root}[stage]()

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('setup','prepare','generate','setup_fixed','features_fixed'):globals()[mode]()
    else:arm_stage(sys.argv[2],mode)
