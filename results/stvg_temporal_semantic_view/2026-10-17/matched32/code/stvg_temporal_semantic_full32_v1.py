"""Prelocked SEM-time four-anchor Direct/OPD continuation, no GT decisions."""
from pathlib import Path
import copy,gc,hashlib,inspect,io,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_temporal_semantic_view32_v1 import BASE as PRIMARY
BASE=PRIMARY/'full';PAPER=ROOT/'artifacts/stvg_paper_t2_b_v1'
def put(p,d,replace=False):write(p,d,mutable=replace)
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    if (BASE/'ACTIVE_EXECUTION.json').exists():assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def prepare():
    import numpy as np,torch
    from PIL import Image
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_evidence_spatial_inputs_v1 import nearest_unique_grid_centers,_video_record
    from scripts.prepare_stvg_short_window_teacher_v1 import render_context_frames
    assert read(PRIMARY/'PRIMARY_ROOT_READBACK.json')['status']=='pass';BASE.mkdir(exist_ok=False);sys.addaudithook(guard)
    for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','BASELINE_BARRIER.json']:(BASE/n).write_bytes((PRIMARY/n).read_bytes())
    jobs=[];tt=[];dec=[];pm={f['case_id']:f for f in read(PRIMARY/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};jm={f['case_id']:f for f in read(PRIMARY/'INPUT_BARRIER.json')['jobs']}
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias'];c,_=load_case(r);f=pm[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);interval=d['arms']['SEM_STUDENT']['interval'];tt.append(dict(case_id=cid,selected_interval=interval,selection='prelocked SEM_STUDENT, no GT',GT_read=False));j=read(ROOT/jm[cid]['path']);assert sha(ROOT/j['source_grid_path'])==j['source_grid_sha256'];frames=np.load(ROOT/j['source_grid_path'],allow_pickle=False);ids=list(map(int,c['frame_ids']));available=list(range(*j['legal_bounds']))
        try:centers=nearest_unique_grid_centers(interval,ids,available)
        except ValueError as e:dec.append(dict(case_id=cid,status='no_legal_four_anchor_support',reason=str(e),GT_read=False));continue
        fids=[x['center_physical_frame_id'] for x in centers];positions=[ids.index(fid) for fid in fids];folder=BASE/'inputs'/cid;folder.mkdir(parents=True);ctx,_=render_context_frames([Image.fromarray(frames[pos]) for pos in positions]);chw=np.ascontiguousarray(ctx.transpose(0,3,1,2));p=folder/'context4.npy';raw=io.BytesIO();np.save(raw,chw,allow_pickle=False);p.write_bytes(raw.getvalue());context=_video_record(str(p.relative_to(ROOT)),raw.getvalue(),chw,[fid/j['source_fps'] for fid in fids],'preserved_unmarked_event_context')
        for slot,(pos,fid) in enumerate(zip(positions,fids)):
            a=frames[pos];p=folder/(str(fid)+'.npy');np.save(p,a,allow_pickle=False);im=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size);jobs.append(dict(job_id=cid+'_'+str(slot),case_id=cid,dataset=r['dataset'],slot=slot,position=pos,physical_frame_id=fid,physical_timestamp=fid/j['source_fps'],image=im,original_image_shape=list(a.shape),original_pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest(),context=copy.deepcopy(context),query_text=j['query_text'],source_case_sha256=c['source_case_sha256'],weight=1.,GT_read=False))
        dec.append(dict(case_id=cid,status='four_fresh_anchors',physical_frame_ids=fids,positions=positions,GT_read=False));del c,frames,d;gc.collect()
    put(BASE/'TEMPORAL_LOCK.json',dict(status='locked',count=32,records=tt,GT_read=False));put(BASE/'INPUT_JOB_LOCK.json',dict(status='locked',count=32,records=jobs,GT_read=False));put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=32,planned_anchors=len(jobs),source_grid_pixels_exact=True,decisions=dec,GT_read=False))
    old=ROOT/'artifacts/stvg_r1_direct_reliable32_v1';oj={j['job_id']:j for j in read(old/'INPUT_JOB_LOCK.json')['records']};reused=[]
    def identity(j):return {k:j[k] for k in ['job_id','case_id','dataset','slot','position','physical_frame_id','physical_timestamp','original_image_shape','original_pixel_sha256','query_text','source_case_sha256']} | {'image_sha256':j['image']['sha256'],'context':{k:v for k,v in j['context'].items() if k not in ['path','bytes']}}
    for j in jobs:
        a=oj.get(j['job_id']);p=old/'teacher'/(j['job_id']+'.json')
        if a is not None and identity(a)==identity(j) and p.exists():reused.append(dict(job_id=j['job_id'],path=str(p.relative_to(ROOT)),sha256=sha(p),complete_input_identity=True,GT_read=False))
    put(BASE/'EXACT_REFERENCE_REUSE_LOCK.json',dict(status='locked',count=len(reused),records=reused,old_root_sha256=sha(old/'R1_FIXED_TEACHER_ROOT.json'),old_runtime_sha256=sha(old/'EXECUTION_RUNTIME001.json'),only_complete_input_identity_no_GT=True,GT_read=False))
def teacher():
    import torch
    from scripts import run_stvg_referent_abc002 as loader
    from scripts.stvg_r1_direct_reliable32_v1 import batch,parse
    verify();sys.addaudithook(guard);loader.BASE=BASE;m,p,device,proof,secs=loader.load_teacher();files=[];qualified=set();jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];reuse={r['job_id']:r for r in read(BASE/'EXACT_REFERENCE_REUSE_LOCK.json')['records']};newcalls=0
    for j in jobs:
        verify();tick=time.perf_counter()
        if j['job_id'] in reuse:
            r=reuse[j['job_id']];assert sha(ROOT/r['path'])==r['sha256'];d=read(ROOT/r['path']);d.update(actual_model_call=False,reused_complete_input_exact=True,reused_output_sha256=r['sha256'],wall_seconds=0.,historical_call_cost_not_zero=True);dest=BASE/'teacher'/(j['job_id']+'.json');put(dest,d);files.append(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest)));qualified.add(j['dataset']);continue
        b,meta=batch(p,j,device);newcalls+=1;put(BASE/'claims'/(j['job_id']+'.json'),dict(status='claimed_once',GT_read=False))
        with torch.inference_mode():o=m.generate(**b,max_new_tokens=256,do_sample=False,use_cache=True,return_dict_in_generate=True,output_attentions=True)
        tokens=o.sequences[0,b['input_ids'].shape[1]:].cpu().tolist();raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();parsed=parse(raw,meta['processed_width'],meta['processed_height'],len(tokens)>=256)
        assert int((b['input_ids']==m.config.image_token_id).sum())==meta['grid_height']*meta['grid_width'];qualified.add(j['dataset']);dest=BASE/'teacher'/(j['job_id']+'.json');put(dest,dict(status='complete',job_id=j['job_id'],case_id=j['case_id'],dataset=j['dataset'],position=j['position'],physical_frame_id=j['physical_frame_id'],input_meta=meta,raw_output=raw,parsed=parsed,truncated=len(tokens)>=256,output_tokens=len(tokens),wall_seconds=time.perf_counter()-tick,clock_proof=proof,GT_read=False));files.append(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest)));put(BASE/'TEACHER_STATUS.json',dict(status='running',done=len(files),total=len(jobs),count=32,GT_read=False),replace=True);del b,o;torch.cuda.empty_cache()
    put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=len(qualified),actual_domains=sorted(qualified),scope='native input/grid only, not semantic labels',GT_read=False));put(BASE/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,new_generations=newcalls,exact_reference_reuses=len(files)-newcalls,model_load_seconds=secs,GT_read=False))
def teacher_root():
    from scripts.stvg_r1_direct_reliable32_v1 import parse
    verify();refs={r['probe_alias']:[] for r in read(BASE/'ROSTER_LOCK.json')['records']};jm={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};dec=[]
    for f in read(BASE/'TEACHER_GLOBAL_BARRIER.json')['files']:
        assert sha(ROOT/f['path'])==f['sha256'];d=read(ROOT/f['path']);j=jm[d['job_id']];meta=d['input_meta'];z=parse(d['raw_output'],meta['processed_width'],meta['processed_height'],d['truncated']);assert z==d['parsed']
        if z['box'] is not None:refs[j['case_id']].append(dict(position=j['position'],physical_frame_id=j['physical_frame_id'],box_cxcywh=z['box'],weight=1.,candidate_id='R1_FIXED_direct',teacher_output_sha256=f['sha256'],GT_read=False))
        dec.append(dict(job_id=j['job_id'],case_id=j['case_id'],status=z['status'],GT_read=False))
    put(BASE/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=[dict(case_id=k,anchors=v) for k,v in refs.items()],decisions=dec,GT_read=False));put(BASE/'TEACHER_ROOT_READBACK.json',dict(status='pass',count=32,GT_read=False))
def setup():
    verify();assert read(BASE/'TEACHER_ROOT_READBACK.json')['status']=='pass';arm=BASE/'arms/FULL';arm.mkdir(parents=True,exist_ok=False)
    for n in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json']:(arm/n).write_bytes((BASE/n).read_bytes())
    put(arm/'SCOPE_CONFIG.json',dict(recipe='lite_fullstep',label='prelocked SEM-time fresh Direct original OPD',GT_read=False));put(arm/'teacher/G32/ROOT_READBACK.json',dict(status='pass',count=32,reference_source_sha256=sha(BASE/'REFERENCE_LOCK.json'),GT_read=False));t=arm/'trials/G32';t.mkdir(parents=True)
    refs={r['case_id']:r['anchors'] for r in read(BASE/'REFERENCE_LOCK.json')['records']};ss={r['case_id']:r for r in read(BASE/'SOURCE_CAPTURE_BARRIER.json')['files']};rows=[dict(case_id=r['probe_alias'],dataset=r['dataset'],source_case_sha256=ss[r['probe_alias']]['source_case_sha256'],anchors=refs[r['probe_alias']],GT_read=False) for r in read(BASE/'ROSTER_LOCK.json')['records']];put(t/'REFERENCE_LOCK.json',dict(status='locked',count=32,records=rows,GT_read=False));(t/'TEMPORAL_LOCK.json').write_bytes((BASE/'TEMPORAL_LOCK.json').read_bytes());d=copy.deepcopy(read(PAPER/'arms/B/trials/G32/DESIGN_LOCK.json'));d.update(count=32,trial_id='G32',scope='semantic_view_full32',cohort='exposed32',family_reuse_from={ds:{m:None for m in v} for ds,v in d['configs'].items()});put(t/'DESIGN_LOCK.json',d)
    paths=[Path(__file__),ROOT/'scripts/stvg_overnight_student_v1.py',ROOT/'scripts/run_stvg_instance_student32_v1.py',ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py',ROOT/'scripts/stvg_soft_ep_coordinate_fit002.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py',ROOT/'scripts/audit_stvg_spatial_lite_fit002.py',ROOT/'scripts/review_stvg_lite_complete002.py',BASE/'EXECUTION_RUNTIME.json',arm/'ROSTER_LOCK.json',arm/'SOURCE_CAPTURE_BARRIER.json',arm/'SCOPE_CONFIG.json',arm/'teacher/G32/ROOT_READBACK.json',t/'REFERENCE_LOCK.json',t/'TEMPORAL_LOCK.json',t/'DESIGN_LOCK.json'];put(t/'RUNTIME.json',dict(status='locked',trial_id='G32',pins={str(p.relative_to(ROOT)):sha(p) for p in paths},protected={},GT_read=False))
def student(mode):
    from scripts import stvg_r1_geometry32_measure_v1 as m,stvg_overnight_student_v1 as s,stvg_pipeline_speed_v1 as speed
    from scripts.resume_stvg_r1_geometry32_dino_v1 import anchors
    verify();anchors();m.BASE=BASE;m.verify=verify;m.configure('FULL')
    if mode=='root':return s.root()
    ns=dict(s.__dict__);ns['speed']=types.SimpleNamespace(BASE=speed.BASE,Engine=lambda label,enabled:speed.Engine('semantic_view_full32_'+label,enabled));exec(compile(inspect.getsource(s.worker),__file__,'exec'),ns);return ns['worker']()
def integrated():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
    from scripts.run_decota_paper_main_v1 import model_for
    from scripts.stvg_temporal_semantic_grounder32_v1 import cached_source
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();sys.addaudithook(guard);from scripts.stvg_temporal_semantic_grounder32_v2 import clean
    clean();torch.set_num_threads(2);t=BASE/'arms/FULL/trials/G32';assert read(t/'ROOT_READBACK.json')['status']=='pass';outputs={r['case_id']:r for r in read(t/'GLOBAL_PREDICTION_BARRIER.json')['files']};fits={r['case_id']:r for r in read(PRIMARY/'FIT_BARRIER.json')['files']};files=[];model=None;src=None
    class Affine(torch.nn.Module):
        def __init__(self,head,theta):super().__init__();self.head=head;self.weight=torch.nn.Parameter(torch.tensor(theta,dtype=torch.float64));assert self.weight.numel()==512
        def forward(self,x):
            y=self.head(x);a=x.detach().cpu().double().numpy()@self.weight.detach().cpu().numpy();return y+torch.from_numpy(a).to(device=y.device,dtype=y.dtype)
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        c,_=load_case(r);cid=r['probe_alias'];source='vidstg' if r['dataset']=='hc2' else 'hcstvg2'
        if source!=src:
            if model is not None:assert state_hash(model.state_dict())==model._fixed_full_state_hash;del model;gc.collect();torch.cuda.empty_cache()
            model=model_for(source);src=source
        f=outputs[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);ff=fits[cid];assert sha(ROOT/ff['path'])==ff['sha256'];td=torch.load(ROOT/ff['path'],map_location='cpu',weights_only=False)['fits']['SEM'];base=cached_source(model,c)
        if d['fits']['M2'] is not None:base.restore(d['fits']['M2']['final_state'])
        head=base.head;base.head=Affine(head,td['theta'])
        with torch.inference_mode():v=base.reference_values(capture=False)
        assert all(torch.equal(a.cpu(),b) for a,b in zip(v['logits'],td['final_logits'])),'Spatial update must not alter Temporal affine head output'
        native=prediction(v['logits'],v['boxes'],c['records'],list(map(int,c['frame_ids'])));assert native['physical_interval']==td['final_interval'];old=d['arms']['TS_M2']
        # The source native prediction helper calls the spatial output boxes_xyxy.
        bb=v['boxes'].detach().cpu().numpy().astype(float);boxes=np.concatenate([bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2],axis=1);assert np.array_equal(boxes,np.asarray(old['boxes']))
        out=dict(case_id=cid,dataset=r['dataset'],arm=dict(boxes=boxes,interval=td['final_interval']),integrated_native_both_branches=True,Spatial_full_logits_and_boxes_exact=True,GT_read=False);f=save_pt(BASE/'integrated'/(cid+'.pt'),out);f['case_id']=cid;files.append(f);base.head=head;del base,c,d,td;gc.collect();torch.cuda.empty_cache()
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False));write(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,actual_integrated_temporal_spatial_readout=True,GT_read=False))
if __name__=='__main__':
    mode=sys.argv[1];student(mode) if mode in ['worker','root'] else globals()[mode]()
