"""Explicit future Qwen adapter and unchanged Student/D2 bridges; never invoked on import."""
from pathlib import Path
import copy,hashlib,inspect,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_qwen35_geometry32_v1 import BASE,PREVIOUS,COMPLEMENT,CHECKPOINT,read,sha,write,execution_gate,verify_scaffold
from scripts.stvg_qwen35_geometry_contract_v1 import messages,parse_final

def render_native_template(processor,job):
    text=processor.apply_chat_template(messages(job),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if not text.endswith('<think>\n\n</think>\n\n'):raise RuntimeError('actual checkpoint template did not close thinking prefix')
    return text

def load_future_model():
    """Requires a separate compatible environment; no Qwen2/R1 fallback or quantization."""
    execution_gate('interface_qualification')
    from scripts.stvg_native_instance_common_v1 import guard
    sys.addaudithook(guard)
    import torch,transformers
    cls=getattr(transformers,'Qwen3_5ForConditionalGeneration',None)
    if cls is None:raise RuntimeError('Qwen3_5ForConditionalGeneration absent; do not upgrade existing R1 runtime in place')
    processor=transformers.AutoProcessor.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False)
    model=cls.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
                             device_map={'':'cuda:0'},low_cpu_mem_usage=True).eval()
    if getattr(model,'is_quantized',False) or model.config.model_type!='qwen3_5':raise RuntimeError('checkpoint architecture/precision substitution')
    if any(p.is_floating_point() and p.dtype not in [torch.bfloat16,torch.float32] for p in model.parameters()):raise RuntimeError('unexpected parameter dtype')
    # Native hybrid-architecture FP32 exceptions/intermediates must be recorded at qualification;
    # asking from_pretrained for BF16 does not authorize casting those components away.
    return model,processor

def prepare_native_batch(processor,job,*,verified_video_adapter):
    """Video metadata API is supplied only after actual-version qualification, never guessed."""
    import numpy as np
    from PIL import Image
    for k in ['image','context']:assert sha(ROOT/job[k]['path'])==job[k]['sha256']
    image=np.load(ROOT/job['image']['path'],allow_pickle=False);video=np.load(ROOT/job['context']['path'],allow_pickle=False)
    assert hashlib.sha256(image.tobytes()).hexdigest()==job['original_pixel_sha256']
    assert image.shape==tuple(job['original_image_shape']) and video.shape==(4,3,448,448)
    text=render_native_template(processor,job)
    if verified_video_adapter is None:raise RuntimeError('actual Qwen video timestamp/processor adapter remains unqualified')
    batch,audit=verified_video_adapter(processor,text,Image.fromarray(image),video,job['context'])
    checks=['anchor_image_identity_verified','native_video_timestamps_verified','resize_crop_pad_transform_verified']
    if not all(audit.get(k) is True for k in checks):raise RuntimeError('native processor qualification missing')
    return batch,audit

def claim_formal_request(job_id):
    """A crashed or invalid request keeps its claim. No automatic output-based retry."""
    execution_gate('teacher');jobs={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']}
    if job_id not in jobs:raise ValueError('unknown planned anchor')
    import os,json
    p=BASE/'formal_claims'/(job_id+'.json');p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(dict(status='claimed_once',job_id=job_id,input_lock_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),GT_read=False),f)
    return jobs[job_id]

def seal_references():
    execution_gate('reference_root');jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];roster=read(BASE/'ROSTER_LOCK.json')['records'];refs={r['probe_alias']:[] for r in roster};files=[];decisions=[]
    model_hash=sha(BASE/'MODEL_SNAPSHOT_LOCK.json');prompt_hash=sha(BASE/'PROMPT_LOCK.json')
    for j in jobs:
        p=BASE/'formal_responses'/(j['job_id']+'.json');d=read(p);claim=read(BASE/'formal_claims'/(j['job_id']+'.json'))
        assert d['job_id']==claim['job_id']==j['job_id'] and d['input_job_sha256']==hashlib.sha256(__import__('json').dumps(j,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert d['model_snapshot_sha256']==model_hash and d['prompt_lock_sha256']==prompt_hash and d['actual_model_call'] is True and d['GT_read'] is False
        parsed=parse_final(d['raw_final_output'],j['job_id'],truncated=d['truncated']);assert d['parsed']==parsed
        if parsed['box_cxcywh'] is not None:refs[j['case_id']].append(dict(position=j['position'],physical_frame_id=j['physical_frame_id'],box_cxcywh=parsed['box_cxcywh'],weight=1.,candidate_id='QWEN_direct',teacher_output_sha256=sha(p),GT_read=False))
        files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size));decisions.append(dict(job_id=j['job_id'],case_id=j['case_id'],status=parsed['status']))
    assert len(files)==128 and len(list((BASE/'formal_claims').glob('*.json')))==128
    write(BASE/'QWEN_REFERENCE_LOCK.json',dict(status='locked',count=32,records=[dict(case_id=k,anchors=v) for k,v in refs.items()],decisions=decisions,GT_read=False))
    write(BASE/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=32,formal_requests=128,files=files,GT_read=False))
    write(BASE/'QWEN_TEACHER_ROOT.json',dict(status='pass',count=32,formal_requests=128,input_sha256=sha(BASE/'INPUT_JOB_LOCK.json'),model_sha256=model_hash,prompt_sha256=prompt_hash,GT_read=False))

def prepare_student():
    execution_gate('student_prepare');assert read(BASE/'QWEN_TEACHER_ROOT.json')['status']=='pass'
    arm=BASE/'arms/QWEN_OPD';arm.mkdir(parents=True,exist_ok=False)
    for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json']:(arm/n).write_bytes((BASE/n).read_bytes())
    write(arm/'SCOPE_CONFIG.json',dict(recipe='lite_fullstep',label='Qwen3.5 direct geometry; unchanged paper OPD',GT_read=False))
    write(arm/'teacher/G32/ROOT_READBACK.json',dict(status='pass',count=32,reference_source_sha256=sha(BASE/'QWEN_REFERENCE_LOCK.json'),scope='Qwen reference preparation only; not Student completion',GT_read=False))
    source={r['case_id']:r for r in read(BASE/'SOURCE_CAPTURE_BARRIER.json')['files']};refs={r['case_id']:r['anchors'] for r in read(BASE/'QWEN_REFERENCE_LOCK.json')['records']}
    rows=[dict(case_id=r['probe_alias'],dataset=r['dataset'],source_case_sha256=source[r['probe_alias']]['source_case_sha256'],anchors=refs[r['probe_alias']],GT_read=False) for r in read(BASE/'ROSTER_LOCK.json')['records']]
    t=arm/'trials/G32';t.mkdir(parents=True);write(t/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=rows,GT_read=False));(t/'TEMPORAL_LOCK.json').write_bytes((BASE/'TEMPORAL_LOCK.json').read_bytes())
    d=copy.deepcopy(read(ROOT/'artifacts/stvg_paper_t2_b_v1/arms/B/trials/G32/DESIGN_LOCK.json'));d.update(count=32,trial_id='G32',scope='Qwen geometry32',cohort='same exposed fixed32',family_reuse_from={ds:{m:None for m in v} for ds,v in d['configs'].items()});write(t/'DESIGN_LOCK.json',d)
    paths=[Path(__file__),ROOT/'scripts/stvg_overnight_student_v1.py',ROOT/'scripts/run_stvg_instance_student32_v1.py',ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py',BASE/'CPU_SCOPE_LOCK.json',BASE/'RUN_AUTHORIZATION.json',BASE/'MODEL_SNAPSHOT_LOCK.json',BASE/'QWEN_TEACHER_ROOT.json',BASE/'QWEN_REFERENCE_LOCK.json',arm/'ROSTER_LOCK.json',arm/'SOURCE_CAPTURE_BARRIER.json',arm/'SCOPE_CONFIG.json',arm/'teacher/G32/ROOT_READBACK.json',t/'REFERENCE_LOCK.json',t/'TEMPORAL_LOCK.json',t/'DESIGN_LOCK.json']
    write(t/'RUNTIME.json',dict(status='locked',trial_id='G32',pins={str(p.relative_to(ROOT)):sha(p) for p in paths},protected={},GT_read=False))

def student(mode):
    execution_gate('student_'+mode)
    from scripts import stvg_r1_geometry32_measure_v1 as m,stvg_pipeline_speed_v1 as speed,stvg_overnight_student_v1 as s
    from scripts.resume_stvg_r1_geometry32_dino_v1 import anchors
    anchors()  # Existing float32 interface tolerance; reference coordinates remain unchanged.
    m.BASE=BASE;m.verify=lambda:execution_gate('student');m.configure('QWEN_OPD')
    if mode=='root':return s.root()
    ns=dict(s.__dict__);ns['speed']=types.SimpleNamespace(BASE=speed.BASE,Engine=lambda label,enabled:speed.Engine('qwen35_geometry32_'+label,enabled));exec(compile(inspect.getsource(s.worker),__file__,'exec'),ns);return ns['worker']()

def propagate():
    execution_gate('propagation');assert read(BASE/'QWEN_TEACHER_ROOT.json')['status']=='pass'
    from scripts import stvg_r1_dino_complement32_v1 as old
    source=inspect.getsource(old.propagate);assert source.count('D1_REFERENCE_LOCK.json')==1
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=lambda:execution_gate('propagation'))
    exec(compile(source.replace('D1_REFERENCE_LOCK.json','QWEN_REFERENCE_LOCK.json'),__file__,'exec'),ns);return ns['propagate']()

def require_prediction_seals_before_score():
    execution_gate('score')
    root=read(BASE/'arms/QWEN_OPD/trials/G32/ROOT_READBACK.json');bar=read(BASE/'arms/QWEN_OPD/trials/G32/GLOBAL_PREDICTION_BARRIER.json');d2=read(BASE/'D2_GLOBAL_PREDICTION_BARRIER.json')
    assert root['status']=='pass' and root['count']==bar['count']==d2['count']==32 and bar['status']==d2['status']=='complete'
    for document in [bar,d2]:
        for f in document['files']:assert sha(ROOT/f['path'])==f['sha256']
    return dict(OPD=bar,D2=d2)
