"""Finite clean corrective-teacher experiment. Capture/teacher workers never load GT."""
import argparse,collections,gc,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/decota_corrective_opd_v1'
MODEL=ROOT/'checkpoints/Qwen3-VL-4B-Instruct'
QUERY='spatial.query_residual'
COMMON='''You are localizing the target of a complete video query. Images 1, 2, 3 are the previous, central and next sampled frames in chronological order; image 4 is a copy of the central frame. They show only a short context, not necessarily the entire queried action. Their original frame IDs are {ids} and times in seconds are {times}. Localize the requested target in the CENTRAL frame. Complete original query: {query}
For a question, localize the entity being asked about, not automatically its grammatical subject: for example, "what is the person holding?" requests the held object. Check both target identity and its full visible boundary. Do not reject a target merely because the described action is not occurring in the center; use the neighboring context and abstain if it is insufficient. Do not invent unseen identity or action evidence.
{mode}
Return only one JSON object with keys "action" and "box". Action 1 supplies a target box; action 2 abstains when evidence cannot reliably determine identity or boundary. {keep}
For action 1, "box" must be four integer relative coordinates [x1,y1,x2,y2], independently normalized to 0..1000 across the full central image width and height, with x1<x2 and y1<y2. For action 0 or 2, "box" must be null. No explanation, markdown, additional fields, multiple boxes or confidence score.'''

def parse(text,original,feedback):
    reason=None
    try:
        x=json.loads(text.strip())
        if not isinstance(x,dict) or set(x)!={'action','box'}:raise ValueError('schema')
        a=x['action']
        if type(a)is not int or a not in [0,1,2]:raise ValueError('action')
        if a==0 and not feedback:raise ValueError('KEEP_without_student')
        if a in [0,2]:
            if x['box']is not None:raise ValueError('unexpected_box')
            return dict(action=a,box=list(original),accepted=a==0,valid=True,reason=None)
        b=x['box']
        if not isinstance(b,list) or len(b)!=4 or any(type(v)is not int for v in b):raise ValueError('coordinates')
        if not all(0<=v<=1000 for v in b):raise ValueError('out_of_bounds')
        if b[0]>=b[2] or b[1]>=b[3]:raise ValueError('invalid_geometry')
        return dict(action=1,box=[(b[0]+b[2])/2000,(b[1]+b[3])/2000,(b[2]-b[0])/1000,(b[3]-b[1])/1000],accepted=True,valid=True,reason=None)
    except (ValueError,TypeError,json.JSONDecodeError) as e:reason=str(e)
    return dict(action=2,box=list(original),accepted=False,valid=False,reason=reason)

def prepare():
    if (OUT/'LOCK.json').exists():return verify()
    parent=ROOT/'artifacts/c1_enabling_tricks_v1/LOCK.json';old=read(parent)
    rows=[{k:r[k] for k in ['key','source','cohort','input','frame_ids','parses']} for r in old['rows']]
    rows.sort(key=lambda r:old['dev'].index(r['key']) if r['key']in old['dev'] else 32+old['validation'].index(r['key']))
    for r in rows:r['split']='development' if r['key']in old['dev'] else 'confirmation'
    assert len(rows)==128 and len({r['source'] for r in rows})==128
    protected=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json','methods/C1_FINAL_RESEARCH_CONFIG.json','methods/C1_TEMPORAL_RESEARCH_STATUS.json','artifacts/c1_fresh_confirmation_v1/vid_train_audit/PROSPECTIVE_SOURCE_RESERVATION.json']
    deps=[Path(__file__),ROOT/'protocols/decota_corrective_opd_v1.md',ROOT/'scripts/run_spatial_ssl_gpu_v1.py',ROOT/'scripts/run_spatial_regression_alignment_v1.py',ROOT/'vg_tta/c1_enabling_tricks_v1.py',ROOT/'vg_tta/c1_luna_tricks_v1.py']+list((ROOT/'methods/decota_final_simplified_v1').glob('*.py'))
    write(OUT/'INPUTS.json',rows)
    write(OUT/'LOCK.json',dict(created=time.time(),user_attachment='./private_authorization_notes/authorization.txt',
          parent_metadata_sha=sha(parent),inputs_sha=sha(OUT/'INPUTS.json'),dev=old['dev'],validation=old['validation'],labels=old['labels'],labels_sha=old['labels_sha'],
          checkpoint=read(ROOT/'methods/C1_FINAL_RESEARCH_CONFIG.json')['source_checkpoint_sha256'],teacher_manifest=sha(MODEL/'DOWNLOAD_MANIFEST.json'),
          pins={str(f):sha(f) for f in deps},protected={str(ROOT/f):sha(ROOT/f) for f in protected},prompt=COMMON,
          max_seconds=7200,peak_bytes=28*2**30,output_bytes=10*2**30,teacher_max_tokens=96,GT_worker=False,DINO_branch=False))
    p=verify();tests=[];b=[.2,.3,.1,.2]
    for text,feedback,a,valid in [(' {"action":0,"box":null}',True,0,True),('{"action":2,"box":null}',True,2,True),('{"action":1,"box":[100,200,300,400]}',True,1,True),('{"action":1,"box":[300,200,100,400]}',True,2,False),('{"action":1,"box":[-1,0,100,100]}',True,2,False),('{"action":0,"box":null}',False,2,False),('```json\n{"action":2,"box":null}\n```',True,2,False)]:
        x=parse(text,b,feedback);assert x['action']==a and x['valid']==valid
        if a in [0,2]:assert x['box']==b
        tests.append(x)
    write(OUT/'PARSER_ENGINEERING.json',dict(status='pass',cases=tests,coordinate_roundtrip_error=0.,KEEP_exact=True,ABSTAIN_no_reference=True,repair=False))
    print('PREPARED32+96',flush=True);return p

def verify():
    p=read(OUT/'LOCK.json');pins=dict(p['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for f,h in {**pins,**p['protected']}.items():assert sha(f)==h,f
    assert sha(OUT/'INPUTS.json')==p['inputs_sha'];return p

def file_for(r):return OUT/'native'/(r['key'].replace(':','_')+'.pt')

def img(frame):
    im=Image.fromarray(frame);scale=min(1.,448/max(im.size));w,h=[max(32,round(v*scale/32)*32) for v in im.size]
    return im.resize((w,h),Image.Resampling.BICUBIC)

def make_jobs(r,frames,z):
    jobs=[];directory=OUT/'images'/r['key'].replace(':','_');directory.mkdir(parents=True,exist_ok=True)
    for ai,pos in enumerate(z['anchors']):
        indexes=[max(0,pos-1),pos,min(len(frames)-1,pos+1)];ims=[img(frames[i]) for i in indexes];box=z['native']['boxes'][pos].tolist()
        xy=np.r_[np.asarray(box[:2])-np.asarray(box[2:])/2,np.asarray(box[:2])+np.asarray(box[2:])/2]
        # Normalization roundtrip is exact up to float arithmetic; no label or clipping.
        rt=np.r_[(xy[:2]+xy[2:])/2,xy[2:]-xy[:2]];assert np.max(np.abs(rt-box))<1e-12
        paths=[]
        for i,im in enumerate(ims):
            f=directory/f'{ai}_{i}.png';im.save(f);paths.append(str(f))
        marked=ims[1].copy();d=ImageDraw.Draw(marked);w,h=marked.size
        d.rectangle(tuple(xy*np.array([w,h,w,h])),outline=(255,64,64),width=3)
        mark=directory/f'{ai}_marked.png';marked.save(mark)
        for mode in ['no_student','feedback']:
            files=paths+[str(mark) if mode=='feedback' else paths[1]]
            student='The fourth image marks the current student prediction in red. It may be WRONG, not ground truth. Its relative xyxy coordinates are '+json.dumps((xy*1000).tolist())+'. You may replace it anywhere in the full image; do not preserve a wrong target just to stay close.'
            prompt=COMMON.format(ids=[r['frame_ids'][i] for i in indexes],times=[round(r['frame_ids'][i]/r['input']['fps'],6) for i in indexes],query=r['input']['caption'],
                  mode=student if mode=='feedback' else 'No student prediction is provided. Independently locate the requested target from the images and query.',
                  keep='Action 0 KEEP is allowed only if BOTH target identity AND full boundary of the supplied student box are acceptable; it copies that box exactly.' if mode=='feedback' else 'Action 0 is unavailable because no current box is supplied. Use action 1 or 2.')
            jobs.append(dict(id=f'{r["key"]}|{ai}|{mode}',key=r['key'],source=r['source'],anchor=ai,position=pos,frame_id=r['frame_ids'][pos],context_positions=indexes,
                        mode=mode,original_box=box,prompt=prompt,images=files,image_shas=[sha(f) for f in files]))
    return jobs

def capture():
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.c1_controlled_corruption_v1 import configuration
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from methods.decota_final_simplified_v1.objectives import prediction,SpatialLoss
    from methods.decota_final_simplified_v1.observations import uniform_positions
    from methods.decota_final_simplified_v1.backbone import query_subject
    from methods.decota_final_simplified_v1.replay import TemporalReplay
    from methods.decota_final_simplified_v1.optim import fit_temporal
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from vg_tta.c1_enabling_tricks_v1 import reinsert
    p=verify();guard=lease();start=time.perf_counter();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();count=collections.Counter();jobs=[]
    try:
        model=model_load('hcstvg1_test');mh=state_hash(model.state_dict());cfg=configuration('vid_source')
        for ordinal,r in enumerate(read(OUT/'INPUTS.json')[:32]):
            assert r['split']=='development';assert sha(r['input']['video_path'])==r['input']['video_sha256']
            frames,ids=decode(r['input']);assert ids==r['frame_ids'];batch,records,s=frozen_forward(model,frames,r)
            assert sum(v.numel() for _,v in s.named)==1792 and torch.count_nonzero(s.initial[QUERY])==0
            native=prediction(s.zero['logits'],s.zero['boxes'],records,ids);apos=uniform_positions(ids,native['indices'],4)
            with torch.no_grad():assert torch.equal(s.values()['boxes'],s.zero['boxes'])
            eng={}
            if ordinal<4:
                initial=s.state();fn=SpatialLoss([dict(position=apos[0],box=native['boxes'][apos[0]].tolist())],s.zero['boxes'])
                opt=torch.optim.Adam([v for _,v in s.named],lr=0.);opt.zero_grad();fn(s.values()['boxes']).backward();opt.step()
                assert all(torch.equal(v,initial[k]) for k,v in s.named)
                empty=SpatialLoss([],s.zero['boxes']);assert empty.empty and float(empty(s.zero['boxes']))==0
                changed=detached(initial);changed[QUERY]=changed[QUERY]+.005*torch.sin(torch.arange(256,device=changed[QUERY].device))
                s.restore(changed)
                with torch.no_grad():value=detached(s.values())
                s.restore(initial);s.restore(changed)
                with torch.no_grad():assert torch.equal(s.values()['boxes'],value['boxes'])
                with query_subject(model,batch,r['parses']['subject']):
                    ll=reinsert(model,batch,ids,records,changed,None,value['boxes'].cpu());count['nonzero_full_reinsertions']+=1
                assert all(torch.equal(a,b) for a,b in zip(ll,s.zero['logits']));s.restore(initial)
                with query_subject(model,batch,r['parses']['subject']):
                    reinsert(model,batch,ids,records,initial,None,native['boxes']);count['source_full_reinsertions']+=1
                eng=dict(zero_lr=True,empty_reference=True,reload=True,nonzero_full_reinsert=True,source_full_reinsert=True,boxes_changed=bool(not torch.equal(value['boxes'],s.zero['boxes'])))
            # Original internal-evidence temporal branch: independent of spatial boxes and of DINO.
            temporal=TemporalReplay(s.head,s.temporal_inputs,s.zero);tt=fit_temporal(temporal,records,cfg,trace=True)
            formal=prediction(tt['shrunk']['logits'],s.zero['boxes'],records,ids)
            z=dict(key=r['key'],source=r['source'],native=native,formal=formal,temporal=detached(tt,'cpu'),initial=detached(s.initial,'cpu'),anchors=apos,
                   frame_ids=ids,pixel_sha=hashlib.sha256(frames.tobytes()).hexdigest(),engineering=eng,GT_used=False,DINO_calls=0,DINO_cache_reads=0)
            f=file_for(r);save(f,z);write(f.with_suffix('.json'),dict(sha=sha(f),lock_sha=sha(OUT/'LOCK.json')));jobs.extend(make_jobs(r,frames,z))
            count['sources']+=1;count['anchors']+=len(apos);count['temporal_updates']+=tt['backwards'];print('CAPTURE',ordinal,r['key'],'anchors',len(apos),'seconds',round(time.perf_counter()-start,1),flush=True)
            assert time.perf_counter()-start<p['max_seconds'] and torch.cuda.max_memory_allocated()<p['peak_bytes']
            del frames,batch,records,s,temporal,tt,z;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh
        write(OUT/'TEACHER_JOBS.json',jobs)
        write(OUT/'CAPTURE_BARRIER.json',dict(files={str(f):sha(f) for f in (OUT/'native').glob('*.pt')},jobs_sha=sha(OUT/'TEACHER_JOBS.json'),counts=dict(count),source_weights_unchanged=True,GT_used=False))
    finally:
        write(OUT/'worker_receipts'/f'capture_{time.time_ns()}.json',dict(seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),counts=dict(count)));guard.close()

def teacher():
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from scripts.run_final_simplification_v1 import lease
    p=verify();bar=read(OUT/'CAPTURE_BARRIER.json');assert sha(OUT/'TEACHER_JOBS.json')==bar['jobs_sha']
    manifest=read(MODEL/'DOWNLOAD_MANIFEST.json');assert sha(MODEL/'DOWNLOAD_MANIFEST.json')==p['teacher_manifest']
    for name,spec in manifest['files'].items():
        if name.endswith('.safetensors'):assert sha(MODEL/name)==spec['lfs']['sha256']
    guard=lease();start=time.perf_counter();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();files=[];count=collections.Counter()
    try:
        model,info=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'];model.eval().requires_grad_(False);pr=AutoProcessor.from_pretrained(MODEL,local_files_only=True);versions=[a._version for a in model.parameters()]
        for j,job in enumerate(read(OUT/'TEACHER_JOBS.json')):
            for f,h in zip(job['images'],job['image_shas']):assert sha(f)==h
            ims=[Image.open(f).convert('RGB') for f in job['images']];content=[dict(type='image') for _ in ims]+[dict(type='text',text=job['prompt'])]
            text=pr.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            inputs=pr(text=[text],images=ims,do_resize=False,return_tensors='pt').to('cuda');t=time.perf_counter()
            with torch.inference_mode():result=model.generate(**inputs,max_new_tokens=p['teacher_max_tokens'],do_sample=False,use_cache=True)
            tokens=result[0,inputs['input_ids'].shape[1]:].cpu().tolist();answer=pr.tokenizer.decode(tokens,skip_special_tokens=True)
            parsed=parse(answer,job['original_box'],job['mode']=='feedback')
            f=OUT/'teacher_results'/(hashlib.sha256(job['id'].encode()).hexdigest()+'.json')
            write(f,dict(id=job['id'],text=answer,tokens=tokens,parsed=parsed,seconds=time.perf_counter()-t,prompt_tokens=inputs['input_ids'].shape[1],generated_tokens=len(tokens),GT_used=False))
            files.append(dict(path=str(f),sha=sha(f)));count[job['mode']]+=1;count['generated_tokens']+=len(tokens)
            if j%16==0:print('TEACHER',j,parsed['action'],parsed['valid'],'seconds',round(time.perf_counter()-start,1),flush=True)
            assert time.perf_counter()-start<p['max_seconds'] and torch.cuda.max_memory_allocated()<p['peak_bytes']
            del inputs,result;gc.collect()
        assert versions==[a._version for a in model.parameters()]
        write(OUT/'TEACHER_BARRIER.json',dict(files=files,counts=dict(count),GT_used=False,model_unchanged=True,worker_seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated()))
    finally:
        write(OUT/'worker_receipts'/f'teacher_{time.time_ns()}.json',dict(seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),counts=dict(count)));guard.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','capture','teacher']);a=ap.parse_args();globals()[a.action]()
