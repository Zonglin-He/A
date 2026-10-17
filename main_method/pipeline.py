"""Phase-separated, resumable main method; GT is opened only after every seal."""
from __future__ import annotations
import collections
import copy
import gc
import hashlib
import json
import re
from pathlib import Path
import time
from .runtime import ROOT,BASE,read,put,sha,resource_guard,status

def path(scope):
    if scope not in ('engineering32','official_clean') and not re.fullmatch(r'official_clean/batches/B\d{4}',scope):
        raise ValueError('unknown fixed cohort')
    return BASE/scope

def save_pt(p,d):
    import torch
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    temp=p.with_name(p.name+'.tmp')
    torch.save(d,temp);temp.replace(p)
    f=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
    put(p.with_suffix('.receipt.json'),dict(status='sealed',**f,GT_read=False))
    return f

def load_pt(f):
    import torch
    p=ROOT/f['path']
    assert sha(p)==f['sha256']
    return torch.load(p,map_location='cpu',weights_only=False)

def prepare():
    from .engine import original_spatial_config
    from scripts.stvg_temporal_marginal_train_fit256_v1 import CFG
    from scripts.stvg_qwen35_geometry_contract_v1 import MODEL_ID,REVISION
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'METHOD_LOCK.json').exists():
        return
    old=read(ROOT/'artifacts/stvg_qwen35_geometry32_v1/ROSTER_LOCK.json')
    full=read(ROOT/'artifacts/stvg_final_table1_v1/INPUT_MANIFEST.json')
    assert len(full['records'])==13785
    assert {d:sum(r['dataset']==d for r in full['records']) for d in ('hc2','vidstg')}==dict(hc2=3482,vidstg=10303)
    for scope,rows in [('engineering32',old['records']),('official_clean',full['records'])]:
        rows=copy.deepcopy(rows)
        for r in rows:
            r['probe_alias']=('HC' if r['dataset']=='hc2' else 'V')+f"{r['query_ordinal']:05d}"
        put(path(scope)/'ROSTER_LOCK.json',dict(status='locked',count=len(rows),records=rows,GT_read=False))
    put(BASE/'METHOD_LOCK.json',dict(status='frozen_before_new_models',teacher=MODEL_ID,revision=REVISION,
        source_checkpoints=dict(hc2=dict(path='checkpoints/TASTVG_VidSTG.pth',sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83'),
            vidstg=dict(path='checkpoints/TASTVG_HCSTVG2.pth',sha256='47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036')),
        Teacher_frozen=True,Teacher_native_non_thinking=True,Teacher_dtype='native BF16 with architecture FP32 exceptions',
        temporal=dict(hypotheses='original I1; I2 descending exact merged interval mass with distinct sampled frames',
            captions='original 8x448 marked video, 160 token cap',comparison='original pairwise A/B/C/D, both display orders, identity-remapped mean',
            reward='r1=p_first-p_second; r2=-r1',fitter=CFG,parameters=512,decoder='exact merged interval marginal MAP'),
        spatial=dict(teacher='same Qwen3.5 Direct',context_frames=4,planned_anchors=4,parameters=1792,
            settings={d:original_spatial_config(d) for d in ('hc2','vidstg')},recipe='lite_fullstep',response_scope='full_grid',
            state='Source and Adam reset independently every Query; no cross-Query inheritance'),
        arms=['Source_MAP','Temporal_only','Spatial_only_native_MAP_evidence','Full'],
        extra_controls=['Source_original_decoder','same_reference_D2'],
        official_count=13785,official_domains=dict(hc2=3482,vidstg=10303),official_parents=dict(hc2=237,vidstg=732),
        target_GT_use='offline evaluation only after complete population prediction seal',
        no_external_Actionness=True,no_DINO=True,no_new_parameter_or_Prompt_search=True,
        robustness=dict(families=['drop','freeze','blur','occlusion','exposure'],physical_time_coverage=[.025,.05,.10],reuse_original_corruption_operators=True),
        budget_analysis=dict(K=[1,2,4,8],fixed_parameters=True),
        writing_and_figure1_out_of_scope=True,CURRENT_changed=False,GT_read=False))
    put(BASE/'STATUS.json',dict(status='prepared',stage='code_cleanup_and_exact_engineering_qualification',actual_new_model_calls=0,GT_read=False),mutable=True)

def cases(scope):
    return read(path(scope)/'ROSTER_LOCK.json')['records']

def once(out, identity, request):
    """Journal a model request before execution, seal its result before reuse.

    An interrupted request without a result is deliberately not retried: its
    actual model-call budget must first be reconciled by the controller owner.
    """
    key=hashlib.sha256(identity.encode()).hexdigest()
    result=out/'requests'/(key+'.json');claim=out/'requests'/(key+'.claim.json')
    if result.exists():
        saved=read(result);assert saved['input_identity']==identity
        return saved['result'],True
    if claim.exists():raise RuntimeError('unsealed model request requires explicit recovery: '+key)
    put(claim,dict(status='claimed_before_model_call',input_identity=identity,GT_read=False))
    value=request();put(result,dict(status='complete',input_identity=identity,result=value,GT_read=False))
    return value,False

def source(scope):
    import numpy as np
    import torch
    from . import engine
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts.stvg_native_instance_common_v1 import guard
    import sys
    sys.addaudithook(guard)
    out=path(scope);files=[];m=None;domain=None;cache=collections.OrderedDict()
    # Complete Source cases may be reused by full identity, never by development score.
    old=read(ROOT/'artifacts/stvg_qwen35_geometry32_v1/ROSTER_LOCK.json')['records']
    oldmap={r['query_key_sha256']:r for r in old}
    for i,r in enumerate(cases(scope)):
        resource_guard();p=out/'source'/(r['probe_alias']+'.pt')
        if p.exists():
            f=read(p.with_suffix('.receipt.json'));assert sha(p)==f['sha256'];files.append(f);continue
        prior=oldmap.get(r['query_key_sha256'])
        if prior and scope=='engineering32':
            from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
            c,_=load_case(prior)
            assert c['parent_sha256']==r['parent_sha256'] and c['query_key_sha256']==r['query_key_sha256']
            f=save_pt(p,c)
        else:
            if domain!=r['dataset']:
                if m is not None:del m;gc.collect();torch.cuda.empty_cache()
                m=engine.model_for(r['dataset']);domain=r['dataset']
            c,base,records,batch,frames,row=engine.capture(m,r,cache)
            f=save_pt(p,c)
            save_pt(out/'prefix'/(r['probe_alias']+'.pt'),dict(views=detached(base.views,'cpu'),source_case_sha256=c['source_case_sha256'],GT_read=False))
            del base,batch,frames;gc.collect()
        files.append(f);status('source_'+scope,i+1,len(cases(scope)),GT_read=False)
    if m is not None:del m;gc.collect();torch.cuda.empty_cache()
    put(out/'SOURCE_BARRIER.json',dict(status='complete',count=len(files),files=files,GT_read=False))

def inputs(scope):
    import numpy as np
    from .engine import row_for
    from scripts.stvg_dino_pool_r1_32_v3 import decode as rgb_decode
    from scripts.stvg_table1_temporal_adapter_v1 import _render_marked_frames
    from scripts.stvg_temporal_marginal_i2_select256_v1 import select
    from scripts.stvg_temporal_marginal_train_fit256_v1 import native
    out=path(scope);files=[]
    for r,s in zip(cases(scope),read(out/'SOURCE_BARRIER.json')['files']):
        p=out/'temporal_inputs'/r['probe_alias']/'JOB.json'
        if p.exists():files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)));continue
        c=load_pt(s);row=row_for(r);available=list(range(*map(int,c['video_bounds'])))
        sel=select(native.offsets(c,c['temporal_hidden']),available)
        hs=sel['hypotheses']
        if len(hs)==2:
            pixels=rgb_decode(c,{int(f) for h in hs for f in h['sampled_physical_frame_ids']})
            for k,h in enumerate(hs):
                h['canonical_id']='Native' if k==0 else 'Alternative'
                a,meta=_render_marked_frames(c,h,h['sampled_physical_frame_ids'],pixels)
                q=p.parent/(str(k)+'.npy');q.parent.mkdir(parents=True,exist_ok=True);np.save(q,a,allow_pickle=False)
                h.update(video_path=str(q.relative_to(ROOT)),video_sha256=sha(q),metadata=meta)
            if hs[0]['video_sha256']==hs[1]['video_sha256'] and hs[0]['metadata']['clip_fps']==hs[1]['metadata']['clip_fps']:
                sel['status']='identical_actual_rendered_observations_no_retry'
        put(p,dict(case_id=r['probe_alias'],dataset=r['dataset'],query_text=row['input']['caption'],selection=sel,source_case_sha256=c['source_case_sha256'],GT_read=False))
        files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    put(out/'TEMPORAL_INPUT_BARRIER.json',dict(status='complete',count=len(files),files=files,GT_read=False))

def temporal_teacher(scope):
    import numpy as np
    import torch
    from . import teacher
    from scripts.stvg_native_instance_common_v1 import guard
    import sys
    sys.addaudithook(guard);m,p=teacher.load();out=path(scope);files=[];qualified=set();calls=0
    for f in read(out/'TEMPORAL_INPUT_BARRIER.json')['files']:
        resource_guard();j=read(ROOT/f['path']);cid=j['case_id'];q=out/'temporal_teacher'/(cid+'.json')
        if q.exists():files.append(dict(path=str(q.relative_to(ROOT)),sha256=sha(q)));continue
        caps=[];result=dict(rewards=[None,None],orders=[],posterior=None)
        if j['selection']['status']=='two_distinct_observations':
            for h in j['selection']['hypotheses']:
                assert sha(ROOT/h['video_path'])==h['video_sha256']
                a=np.load(ROOT/h['video_path'],allow_pickle=False);b,text=teacher.caption_batch(p,a,h['metadata']['clip_fps'])
                identity='caption160:'+h['video_sha256']+':'+repr(h['metadata']['clip_fps'])
                cap,reused=once(out,identity,lambda:teacher.generate(m,p,b,160));calls+=not reused;caps.append(cap)
                if j['dataset'] not in qualified:
                    again,reused=once(out,'qualification8:'+identity,lambda:teacher.generate(m,p,b,8));calls+=not reused
                    assert again['token_ids']==cap['token_ids'][:len(again['token_ids'])]
                    qualified.add(j['dataset'])
            if all(c['raw'] and not c['truncated'] for c in caps):
                texts=[c['raw'] for c in caps]
                identity='pairwise_both_orders:'+json.dumps([j['query_text'],texts],ensure_ascii=False)
                result,reused=once(out,identity,lambda:teacher.pairwise(m,p,j['query_text'],texts));calls+=0 if reused else 2
        put(q,dict(status='complete',case_id=cid,captions=caps,**result,GT_read=False))
        files.append(dict(path=str(q.relative_to(ROOT)),sha256=sha(q)));status('temporal_teacher_'+scope,len(files),len(cases(scope)),actual_calls=calls,GT_read=False)
    del m,p;gc.collect();torch.cuda.empty_cache()
    put(out/'TEMPORAL_TEACHER_BARRIER.json',dict(status='complete',count=len(files),files=files,new_calls=calls,GT_read=False))
    put(out/'TEMPORAL_GPU_QUALIFICATION.json',dict(status='pass' if qualified=={'hc2','vidstg'} else 'incomplete',count=len(qualified),actual_8token_prefix_exact=True,GT_read=False))

def temporal_fit(scope):
    from .engine import temporal
    out=path(scope);files=[]
    for r,s,j,t in zip(cases(scope),read(out/'SOURCE_BARRIER.json')['files'],read(out/'TEMPORAL_INPUT_BARRIER.json')['files'],read(out/'TEMPORAL_TEACHER_BARRIER.json')['files']):
        q=out/'temporal_fits'/(r['probe_alias']+'.pt')
        if q.exists():files.append(read(q.with_suffix('.receipt.json')));continue
        c=load_pt(s);jj=read(ROOT/j['path']);tt=read(ROOT/t['path'])
        result=temporal(c,jj['selection']['hypotheses'],tt['rewards'])
        files.append(save_pt(q,dict(case_id=r['probe_alias'],fit=result,GT_read=False)))
        status('temporal_fit_'+scope,len(files),len(cases(scope)),GT_read=False)
    put(out/'TEMPORAL_FIT_BARRIER.json',dict(status='complete',count=len(files),files=files,GT_read=False))

def replay(model,c,r,out):
    import torch
    from .engine import capture
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from scripts.stvg_pipeline_speed_v1 import to_device
    from scripts.stvg_temporal_semantic_grounder32_v1 import cached_source
    p=out/'prefix'/(r['probe_alias']+'.pt')
    if p.exists():
        v=torch.load(p,map_location='cpu',weights_only=False);assert v['source_case_sha256']==c['source_case_sha256']
        base=SpatialReplay(model,to_device(v['views'],next(model.parameters()).device),len(c['frame_ids']))
    else:
        old=ROOT/'artifacts/stvg_pipeline_speed_v1/source_features'/(c['case_id']+'.pt')
        if old.exists():base=cached_source(model,c)
        else:
            fresh,base,*rest=capture(model,r,collections.OrderedDict())
            assert fresh['pixel_sha256']==c['pixel_sha256']
    assert torch.equal(base.zero['boxes'].cpu(),c['zero']['boxes'])
    assert all(torch.equal(a.cpu(),b) for a,b in zip(base.zero['logits'],c['zero']['logits']))
    return base

def temporal_readout(scope):
    import torch
    from .engine import model_for
    from scripts.stvg_temporal_marginal_decode_util256_v1 import decode
    from vg_tta.stvg_native_head_residual32_v1 import TemporalHeadResidual
    out=path(scope);files=[];m=None;domain=None
    for r,s,f in zip(cases(scope),read(out/'SOURCE_BARRIER.json')['files'],read(out/'TEMPORAL_FIT_BARRIER.json')['files']):
        q=out/'temporal_predictions'/(r['probe_alias']+'.pt')
        if q.exists():files.append(read(q.with_suffix('.receipt.json')));continue
        resource_guard();c=load_pt(s);fit=load_pt(f)['fit']
        if domain!=r['dataset']:
            if m is not None:del m;gc.collect();torch.cuda.empty_cache()
            m=model_for(r['dataset']);domain=r['dataset']
        base=replay(m,c,r,out);head=base.head;base.head=TemporalHeadResidual(head,fit['theta'])
        with torch.no_grad():v=base.reference_values(capture=False)
        assert torch.equal(v['boxes'].cpu(),c['zero']['boxes'])
        assert all(torch.equal(a.cpu(),b) for a,b in zip(v['logits'],fit['final_logits']))
        files.append(save_pt(q,dict(case_id=r['probe_alias'],interval=decode(c['records'],v['logits'])[0],
            source_interval=decode(c['records'],c['zero']['logits'])[0],actual_logits=[x.cpu() for x in v['logits']],GT_read=False)))
        del base,v;gc.collect();status('temporal_readout_'+scope,len(files),len(cases(scope)),GT_read=False)
    if m is not None:del m;gc.collect();torch.cuda.empty_cache()
    put(out/'TEMPORAL_PREDICTION_BARRIER.json',dict(status='complete',count=len(files),files=files,actual_head_and_Source_boxes_exact=True,GT_read=False))

def spatial_inputs(scope):
    import numpy as np
    from PIL import Image
    from .engine import row_for
    from scripts.stvg_dino_pool_r1_32_v3 import decode as rgb_decode
    from scripts.stvg_evidence_spatial_inputs_v1 import nearest_unique_grid_centers
    from scripts.prepare_stvg_short_window_teacher_v1 import render_context_frames
    out=path(scope);files=[]
    for r,s,t in zip(cases(scope),read(out/'SOURCE_BARRIER.json')['files'],read(out/'TEMPORAL_PREDICTION_BARRIER.json')['files']):
        q=out/'spatial_inputs'/r['probe_alias']/'JOBS.json'
        if q.exists():files.append(dict(path=str(q.relative_to(ROOT)),sha256=sha(q)));continue
        c=load_pt(s);temp=load_pt(t);row=row_for(r);roles={}
        for role,iv in [('Spatial_only',temp['source_interval']),('Full',temp['interval'])]:
            try:
                centers=nearest_unique_grid_centers(iv,c['frame_ids'],range(*map(int,c['video_bounds'])))
            except ValueError as e:
                roles[role]=dict(interval=iv,jobs=[],reason=str(e));continue
            fids=[x['center_physical_frame_id'] for x in centers];rgb=rgb_decode(c,set(fids))
            arr,meta=render_context_frames([Image.fromarray(rgb[x]) for x in fids]);arr=np.ascontiguousarray(arr.transpose(0,3,1,2))
            p=q.parent/role/'context.npy';p.parent.mkdir(parents=True,exist_ok=True);np.save(p,arr,allow_pickle=False)
            times=[x/float(c['input']['fps']) for x in fids];gaps=[b-a for a,b in zip(times,times[1:])]
            context=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),fps=1/(sum(gaps)/len(gaps)),actual_timestamps_seconds=times,
                encoder_uniform_spacing_approximation=max(gaps)-min(gaps)>1e-12)
            jobs=[]
            for slot,fid in enumerate(fids):
                p=q.parent/role/(str(slot)+'.npy');np.save(p,rgb[fid],allow_pickle=False)
                jobs.append(dict(job_id=r['probe_alias']+'_'+str(slot),case_id=r['probe_alias'],dataset=r['dataset'],
                    slot=slot,position=list(map(int,c['frame_ids'])).index(fid),physical_frame_id=fid,
                    physical_timestamp=fid/float(c['input']['fps']),query_text=row['input']['caption'],context=context,
                    image=dict(path=str(p.relative_to(ROOT)),sha256=sha(p)),GT_read=False))
            roles[role]=dict(interval=iv,jobs=jobs)
        put(q,dict(case_id=r['probe_alias'],roles=roles,GT_read=False));files.append(dict(path=str(q.relative_to(ROOT)),sha256=sha(q)))
    put(out/'SPATIAL_INPUT_BARRIER.json',dict(status='complete',count=len(files),files=files,GT_read=False))

def spatial_teacher(scope):
    import numpy as np
    import torch
    from . import teacher
    out=path(scope);m,p=teacher.load();files=[];cache={};calls=0
    for f in read(out/'SPATIAL_INPUT_BARRIER.json')['files']:
        resource_guard();j=read(ROOT/f['path']);q=out/'spatial_teacher'/(j['case_id']+'.json')
        if q.exists():files.append(dict(path=str(q.relative_to(ROOT)),sha256=sha(q)));continue
        roles={}
        for role,v in j['roles'].items():
            refs=[];responses=[]
            for job in v['jobs']:
                from scripts.stvg_qwen35_geometry_contract_v1 import prompt
                for kind in ('image','context'):assert sha(ROOT/job[kind]['path'])==job[kind]['sha256']
                # Full identity includes the actual prompt, role-independent only when all evidence agrees.
                key=hashlib.sha256((prompt(job)+job['image']['sha256']+job['context']['sha256']+json.dumps(job['context']['fps'])).encode()).hexdigest()
                if key in cache:d=copy.deepcopy(cache[key]);reused=True
                else:
                    image=np.load(ROOT/job['image']['path'],allow_pickle=False);ctx=np.load(ROOT/job['context']['path'],allow_pickle=False)
                    d,reused=once(out,'Direct:'+key,lambda:teacher.direct(m,p,job,image,ctx))
                    cache[key]=copy.deepcopy(d);calls+=not reused
                responses.append(dict(job_id=job['job_id'],result=d,exact_reuse=reused,input_identity=key))
                b=d['parsed']['box_cxcywh']
                if b is not None:refs.append(dict(position=job['position'],physical_frame_id=job['physical_frame_id'],box=b,weight=1.))
            roles[role]=dict(interval=v['interval'],anchors=refs,responses=responses)
        put(q,dict(case_id=j['case_id'],roles=roles,GT_read=False));files.append(dict(path=str(q.relative_to(ROOT)),sha256=sha(q)))
        status('spatial_teacher_'+scope,len(files),len(cases(scope)),actual_calls=calls,GT_read=False)
    del m,p;gc.collect();torch.cuda.empty_cache()
    put(out/'SPATIAL_TEACHER_BARRIER.json',dict(status='complete',count=len(files),files=files,new_calls=calls,GT_read=False))

def spatial_student(scope):
    import numpy as np
    import torch
    from .engine import model_for,spatial
    from .runtime import AtomicWriter
    from scripts.stvg_pipeline_speed_v1 import copy_cpu,full_prefix_readout
    from scripts.run_stvg_table1_student004 import stable_fit_key
    out=path(scope);files=[];m=None;domain=None;writer=AtomicWriter()
    try:
        for r,s,t,g in zip(cases(scope),read(out/'SOURCE_BARRIER.json')['files'],read(out/'TEMPORAL_PREDICTION_BARRIER.json')['files'],read(out/'SPATIAL_TEACHER_BARRIER.json')['files']):
            q=out/'predictions'/(r['probe_alias']+'.pt')
            if q.exists():files.append(read(q.with_suffix('.receipt.json')));continue
            resource_guard();c=load_pt(s);temp=load_pt(t);geom=read(ROOT/g['path'])
            if domain!=r['dataset']:
                if m is not None:del m;gc.collect();torch.cuda.empty_cache()
                m=model_for(r['dataset']);domain=r['dataset']
            base=replay(m,c,r,out);native=np.asarray(c['native_boxes_xyxy_normalized']).copy()
            arms=dict(Source_MAP=dict(boxes=native,interval=temp['source_interval']),
                Source_original_decoder=dict(boxes=native,interval=list(c['Native_interval'])),
                Temporal_only=dict(boxes=native,interval=temp['interval']))
            fits={}
            for role,v in geom['roles'].items():
                def checkpoint(d,role=role):
                    writer.save(out/'checkpoints'/r['probe_alias']/role/('step'+str(d.get('next_step',len(d.get('path',[]))))+'.pt'),copy_cpu(d))
                fit=spatial(base,c,v['anchors'],stable_fit_key(r['dataset'],r['query_key_sha256']),checkpoint)
                from scripts.audit_stvg_spatial_lite_fit002 import fit_audit
                fit_audit(fit,dict(anchors=v['anchors']),'lite_fullstep')
                actual=full_prefix_readout(base,list(map(int,c['frame_ids'])),c['records'],fit['final_state'])
                b=actual['boxes'].detach().cpu().numpy().astype(np.float64)
                assert np.allclose(b,fit['final_cached_decoder_boxes'].cpu().numpy(),rtol=0,atol=1e-7)
                xyxy=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)
                arm='Spatial_only_native_MAP_evidence' if role=='Spatial_only' else 'Full'
                arms[arm]=dict(boxes=xyxy,interval=v['interval'])
                fits[role]=fit
                save_pt(out/'spatial_fits'/r['probe_alias']/(role+'.pt'),dict(fit=fit,GT_read=False))
                if role=='Full':
                    arms['same_reference_D2']=dict(boxes=same_reference_d2(native,c['frame_ids'],v['anchors'],v['interval']),interval=v['interval'])
            assert arms['Full']['interval']==arms['Temporal_only']['interval']
            assert arms['Spatial_only_native_MAP_evidence']['interval']==arms['Source_MAP']['interval']
            files.append(save_pt(q,dict(case_id=r['probe_alias'],dataset=r['dataset'],arms=arms,
                temporal_fit_sha256=t['sha256'],temporal_prediction_interval_exact=True,
                observed_positions={k:[a['position'] for a in v['anchors']] for k,v in geom['roles'].items()},GT_read=False)))
            del base,fits,c;gc.collect();status('student_'+scope,len(files),len(cases(scope)),GT_read=False)
        writer.close()
    except BaseException:
        writer.close();raise
    finally:
        if m is not None:del m;gc.collect();torch.cuda.empty_cache()
    put(out/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=len(files),files=files,all_predictions_sealed_before_GT=True,GT_read=False))
    put(out/'ROOT_READBACK.json',dict(status='pass',count=len(files),actual_complete_decoder=True,temporal_invariants_exact=True,GT_read=False))

def same_reference_d2(source,frame_ids,anchors,interval):
    """The existing center/log-size Source-residual physical-time control."""
    import numpy as np
    source=np.asarray(source);out=source.copy()
    if not anchors:return out
    wh=source[:,2:]-source[:,:2]
    assert np.isfinite(source).all() and (wh>0).all()
    c=np.c_[(source[:,:2]+source[:,2:])/2,np.log(wh)]
    pp=np.array([a['position'] for a in anchors]);aa=np.array([a['box'] for a in anchors],float)
    ids=np.asarray(frame_ids,float);res=np.c_[aa[:,:2],np.log(aa[:,2:])]-c[pp]
    order=np.argsort(ids[pp]);xp=ids[pp][order];res=res[order]
    delta=np.stack([np.interp(ids,xp,res[:,k]) for k in range(4)],1)
    z=c+delta;wh=np.exp(z[:,2:]);bb=np.c_[z[:,:2]-wh/2,z[:,:2]+wh/2]
    # Preserve the historical D2 boundary convention exactly as a named control.
    active=(ids>=interval[0])&(ids<=interval[1]);out[active]=bb[active]
    assert np.isfinite(out).all()
    for a in anchors:
        x,y,w,h=a['box'];assert np.allclose(out[a['position']],[x-w/2,y-h/2,x+w/2,y+h/2],rtol=0,atol=1e-7)
    return out

def score(scope):
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    out=path(scope);bar=read(out/'GLOBAL_PREDICTION_BARRIER.json');rr=cases(scope)
    assert bar['status']=='complete' and bar['count']==len(rr)==read(out/'ROOT_READBACK.json')['count']
    source_files=read(out/'SOURCE_BARRIER.json')['files']
    assert len(rr)==len(bar['files'])==len(source_files)
    assert len({r['probe_alias'] for r in rr})==len(rr)
    cs={};seen=set()
    for r,s,f in zip(rr,source_files,bar['files']):
        c=load_pt(s);d=load_pt(f);cid=r['probe_alias']
        assert c['query_key_sha256']==r['query_key_sha256'] and c['parent_sha256']==r['parent_sha256']
        assert d['case_id']==cid and d['dataset']==r['dataset'] and cid not in seen
        assert c['dataset']==r['dataset'];seen.add(cid);cs[cid]=c
    assert seen=={r['probe_alias'] for r in rr}
    targets,bindings=truth._truths_after_seal(cs);off,ind,_=truth._metric_functions();rows=[]
    for r,f in zip(rr,bar['files']):
        d=load_pt(f)
        for arm,a in d['arms'].items():
            m,_=metric._score_student_arm(a,cs[r['probe_alias']],targets[r['probe_alias']],set(),off,ind)
            m.update(R_vIoU_at_0p3=float(m['m_vIoU']>.3),R_vIoU_at_0p5=float(m['m_vIoU']>.5))
            rows.append(dict(case_id=r['probe_alias'],dataset=r['dataset'],parent_sha256=r['parent_sha256'],arm=arm,metrics=m))
    summary,paired=subset_aggregate(rows)
    put(out/'POSTSEAL_RESULTS.json',dict(status='complete',count=len(rr),rows=rows,summary=summary,paired=paired,
        GT_bindings=bindings,GT_read=True,official_population=(scope=='official_clean'),engineering_development=(scope=='engineering32')))
    put(out/'CPU_COMPLETION.json',dict(status='complete',count=len(rr),GT_read=True,result_sha256=sha(out/'POSTSEAL_RESULTS.json')))
