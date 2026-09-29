"""S0 setup/expert/TA workers; GT is absent from all inference paths."""
import sys,time,gc,traceback,subprocess,shutil,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/tastvg_spatial_expansion_s0_v1'
OLD=ROOT/'artifacts/tastvg_transient_c25_c3_c06_v1';D=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2'
MODEL=ROOT/'checkpoints/Sa2VA-4B'
CONDS=['clean']+[x+'_5' for x in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure']]


def prepare():
    p=read(OLD/'LOCK.json');bar=read(OLD/'CAPTURE_BARRIER.json');rows=p['rows'][:16]
    files={f"capture/{c}/{r['ordinal']:03}.pt":bar['files'][f"capture/{c}/{r['ordinal']:03}.pt"] for r in rows for c in CONDS}
    paths=['protocols/tastvg_spatial_expansion_s0_v1.md','vg_tta/tastvg_spatial_expansion_s0_v1.py','vg_tta/tastvg_evidence_capture_v1.py','vg_tta/tastvg_causal_round2_v1.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=rows,conditions=CONDS,capture_files=files,pins={f:sha(ROOT/f) for f in paths},steps=3,relative_step=.004,expert_frames=5,model_revision='3fee777d49ee9276eac51ea3e5f9b69e81d09be6',cap_seconds=7200,GT_read=False,time=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    return p


def observation(row,cond,frames):
    from vg_tta.tastvg_deployment_corruption_v2 import apply_burst
    from scripts.c1_controlled_corruption_v1 import pixelhash
    rel=f"capture/{cond}/{row['ordinal']:03}.pt";assert sha(OLD/rel)==read(OUT/'LOCK.json')['capture_files'][rel];x=load(OLD/rel)
    if cond=='clean':shifted=frames
    else:
        c=load(D/'c05'/cond/f"{row['ordinal']:03}.pt")
        shifted=apply_burst(frames,row['input'],c['generation'],cond.rsplit('_',1)[0],row['source'])
    assert pixelhash(shifted)==x['pixel_sha']
    return shifted,x


def run(stage,limit=0):
    tick=time.monotonic();p=verify();done=0;state='failed';failure=None;lease=None
    prior=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'))
    try:
        import numpy as np,torch
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        if stage=='expert':
            from PIL import Image
            from transformers import AutoModel,AutoTokenizer
            from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            receipt=read(MODEL/'DOWNLOAD_RECEIPT.json');assert receipt['revision']==p['model_revision']
            code_receipt=read(MODEL/'OFFICIAL_CODE_RECEIPT.json');assert code_receipt['revision']==p['model_revision']
            for f,h in code_receipt['files'].items():assert sha(MODEL/f)==h['sha256']
            for f in receipt['files']:assert sha(MODEL/f['file'])==f['sha256']
            model,loading=AutoModel.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True,local_files_only=True,output_loading_info=True)
            assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'],loading
            model=model.eval().cuda().requires_grad_(False)
            tokenizer=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
            model.preparing_for_generation(tokenizer,max_new_tokens=256,torch_dtype=torch.bfloat16)
            holes=[float(m.fill_hole_area) for m in model.modules() if hasattr(m,'fill_hole_area')];assert holes and all(v==0 for v in holes),holes
            versions={n:__import__(n).__version__ for n in ['torch','transformers','tokenizers','peft','accelerate']}
            setup=OUT/'EXPERT_SETUP.json'
            if not setup.exists():write(setup,dict(versions=versions,revision=p['model_revision'],official_code_receipt=code_receipt,files=receipt['files'],code={f.name:sha(f) for f in MODEL.glob('*.py')},loading_info=loading,fill_hole_area=holes,attention='eager',dtype='bfloat16',max_new_tokens=256))
        else:
            from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
            from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
            from methods.decota_final_simplified_v1.tensors import state_hash
            from vg_tta.tastvg_evidence_capture_v1 import capture
            from vg_tta.tastvg_temporal_fourarm_v1 import reinsert
            from vg_tta.tastvg_spatial_expansion_s0_v1 import expand
            install_clean_loader()
            from scripts.run_spatial_regression_alignment_v1 import model_load
            model=model_load('hcstvg1_test');model.eval().requires_grad_(False)
            mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256']
            if stage=='ta':eb=read(OUT/'EXPERT_BARRIER.json');hb=read(OUT/'H_BARRIER.json')
        for row in p['rows']:
            frames=None;cache={}
            for cond in CONDS:
                f=OUT/stage/cond/f"{row['ordinal']:03}.pt"
                if f.with_suffix('.json').exists():
                    assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1
                    if stage=='expert':e=load(f);cache[e['sample_sha256']]=str(f.relative_to(OUT))
                    continue
                if limit and done>=limit:break
                assert prior+time.monotonic()-tick<p['cap_seconds']-20
                assert shutil.disk_usage(ROOT).free>8*2**30
                if frames is None:frames,ids=decode(row['input']);assert ids==row['frame_ids']
                shifted,x=observation(row,cond,frames);start=time.monotonic()
                if stage=='expert':
                    positions=np.rint(np.linspace(0,len(ids)-1,5)).astype(int).tolist();samples=[shifted[i] for i in positions]
                    samplehash=hashlib.sha256(b''.join(im.tobytes() for im in samples)).hexdigest();reuse=cache.get(samplehash)
                    if reuse:
                        old=load(OUT/reuse);result={**old,'condition':cond,'pixel_sha256':x['pixel_sha'],'reused_from':reuse,'seconds':time.monotonic()-start}
                    else:
                        prompt='<image>Please segment the object described by: '+row['input']['caption'].rstrip('.')+'.'
                        with torch.inference_mode():raw=model.predict_forward(video=[Image.fromarray(im) for im in samples],text=prompt,tokenizer=tokenizer)
                        mm=raw['prediction_masks'];valid,boxes=mask_boxes(mm[0] if len(mm) else None,positions,len(ids))
                        result=dict(parent=row['ordinal'],condition=cond,positions=positions,frame_ids=[ids[i] for i in positions],masks=mm[0] if len(mm) else None,mask_count=len(mm),prediction_text=raw['prediction'],valid=valid,boxes=boxes,sample_sha256=samplehash,pixel_sha256=x['pixel_sha'],reused_from=None,seconds=time.monotonic()-start,GT_read=False)
                    cache[samplehash]=str(f.relative_to(OUT));info=f"valid={int(result['valid'].sum())} reused={reuse is not None}"
                elif stage=='capture':
                    native=x['native'];req=dict(input=row['input'],frame_ids=ids,subject=row['parses']['subject'],native_boxes=native['boxes'],native_logits=native['logits'])
                    result=capture(model,shifted,req);result['frame_ids']=ids;result['pixel_sha256']=x['pixel_sha'];info='native_exact=True'
                else:
                    rel=f"expert/{cond}/{row['ordinal']:03}.pt";assert sha(OUT/rel)==eb['files'][rel];expert=load(OUT/rel);assert expert['pixel_sha256']==x['pixel_sha']
                    native=x['native'];req=dict(input=row['input'],frame_ids=ids,subject=row['parses']['subject'],native_boxes=native['boxes'],native_logits=native['logits'])
                    hrel=f"capture/{cond}/{row['ordinal']:03}.pt";assert sha(OUT/hrel)==hb['files'][hrel]
                    data=load(OUT/hrel);assert data['pixel_sha256']==x['pixel_sha'];data=device_tree(data,'cuda')
                    result,fields,ev=expand(model,data,expert)
                    if row['ordinal']==p['rows'][0]['ordinal'] and cond in ['clean','frame_drop_5']:result['reinsertion']=reinsert(model,shifted,row,fields,ev)
                    result.update(parent=row['ordinal'],condition=cond,pixel_sha256=x['pixel_sha'],model_state_sha256=mh,exact_historical_native=True,GT_read=False,seconds=time.monotonic()-start)
                    del data,fields,ev;info='loss='+str(round(result['diagnostics'][-1]['loss_after'],4))
                save(f,result);write(f.with_suffix('.json'),dict(sha256=sha(f)));done+=1
                status(OUT/'STATUS.json',dict(status='running',stage=stage,done=done,total=96,seconds=time.monotonic()-tick));print(stage,done,96,cond,info,'seconds',round(time.monotonic()-start,2),flush=True)
                del result,x,shifted;gc.collect();torch.cuda.empty_cache()
            if limit and done>=limit:break
        if stage!='expert':assert state_hash(model.state_dict())==mh
        verify();state='completed' if done==96 else 'smoke_complete'
        if done==96:write(OUT/({'expert':'EXPERT_BARRIER.json','capture':'H_BARRIER.json','ta':'PREDICTION_BARRIER.json'}[stage]),dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/stage).rglob('*.pt')},cells=96,GT_read=False,time=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        result=dict(status=state,stage=stage,done=done,seconds=time.monotonic()-tick,prior_seconds=prior,failure=failure,runner_sha256=sha(Path(__file__)),time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',result);status(OUT/'STATUS.json',result)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','capture','expert','ta']);a.add_argument('--limit',type=int,default=0);args=a.parse_args();prepare() if args.stage=='prepare' else run(args.stage,args.limit)
