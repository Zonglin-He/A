"""F21 prepare/capture/predict barriers, no GT readers in this executable."""
import argparse,collections,fcntl,hashlib,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
from scripts.run_decota_decision_v1 import plan as oldplan,allrows,file,checked
from vg_tta.decota_spatial_extension_v1 import parse_context,pool_gate
OUT=ROOT/'artifacts/decota_spatial_extension_v1'
F20=ROOT/'artifacts/decota_readout_interfaces_v1'
WATCH=['vidstg_test:004538','vidstg_test:009379','vidstg_test:005972','hcstvg1_test:000194','hcstvg1_test:000822']


def verify_old():
    z=read(F20/'FINAL_SEAL.json')
    for field in ['code_hashes','result_hashes']:
        for f,h in z[field].items():assert sha(ROOT/f)==h,(field,f)
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==z['current_method_sha256']
    return z


def dest(stage,key):return OUT/stage/(key.replace(':','_')+'.pt')
def rank(k):return hashlib.sha256(('F21:'+k).encode()).hexdigest()


def prepare():
    from scripts.run_stvg_fullscale_v1 import query_parser
    from vg_tta.tg_spatial_tta_v1 import visual_query
    from methods.decota_refine_uniform_v1.api import uniform_positions
    verify_old();p=oldplan();old=allrows(p);lookup={r['key']:r for r in old};watch=[lookup[k] for k in WATCH]
    excluded_sources={r['source'] for r in watch};excluded_hashes={r['input']['video_sha256'] for r in watch}
    main=[dict(r,role='extension',supplemented=False) for r in old if r['split']=='extension' and r['source'] not in excluded_sources and r['input']['video_sha256'] not in excluded_hashes]
    full=read(ROOT/'artifacts/stvg_fullscale_diagnostics_v1/lock.json');excluded=[]
    blocked={r['source'] for r in old};hashes={r['input']['video_sha256'] for r in old}
    for cohort in p['cohorts']:
        need=32-sum(r['key'].startswith(cohort+':') for r in main)
        if not need:continue
        grouped=collections.defaultdict(list)
        for r in full['rows'][cohort]:
            q=r['input']
            if not r['input_unavailable'] and q['source'] not in blocked and q['video_sha256'] not in hashes:grouped[q['source']].append(r)
        for src in sorted(grouped,key=rank):
            selected=None
            for r in sorted(grouped[src],key=lambda x:rank(x['key'])):
                cache=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/method/tastvg'/cohort/(r['key'].split(':')[1]+'.pt')
                if not cache.exists():excluded.append(dict(key=r['key'],reason='missing_cache'));continue
                x=load(cache);audit=x['temporal_audit'];loss=[audit['audit']['initial_loss']]+[z['loss_after'] for z in audit['steps']]
                best=min(range(len(loss)),key=lambda j:(loss[j],j))
                if best not in (0,5):excluded.append(dict(key=r['key'],reason='B_intermediate_not_saved',best=best));continue
                selected=dict(r,source=src,role='extension',split='extension_replenishment',outer_fold=-2,supplemented=True,
                    cache_path=str(cache),cache_sha256=sha(cache),cached_B_step=best,cached_losses=loss)
                break
            if selected:
                main.append(selected);blocked.add(src);hashes.add(selected['input']['video_sha256']);need-=1
            if not need:break
        assert need==0
    rows=main+[dict(r,role='watched',supplemented=False) for r in watch]
    assert len(rows)==69 and len({r['source'] for r in rows})==69
    parser=query_parser();selection=read(ROOT/'artifacts/decota_decision_v1/spatial_selection.json');coverage=collections.Counter()
    for r in rows:
        cohort=r['key'].split(':')[0];r['cohort']=cohort
        r['context_parse']=parse_context(parser,r['input']['caption'])
        if r['supplemented']:
            r['original_parse']=visual_query(parser,r['input']['caption']);c=load(r['cache_path']);ids=c['frame_ids']
            r['B']=c['predictions']['frozen' if r['cached_B_step']==0 else 'temporal_only']['indices']
            r['base_cache']=r['cache_path'];r['base_cache_sha256']=r['cache_sha256']
        else:
            r['original_parse']=r['parser_audit']['original'];x=checked(file('temporal',r['key']));ids=x['frame_ids'];r['B']=x['candidates'][0]['indices']
            r['base_cache']=str(file('temporal',r['key']));r['base_cache_sha256']=sha(r['base_cache'])
        r['positions']=uniform_positions(ids,r['B'],8);r['frame_ids']=ids
        config=selection['configurations'][cohort+':-1'];r['weight']=config['choices']['reference']['weight']
        assert r['source'] not in config['train_sources']
        coverage[cohort+'/'+r['role']+'/'+r['context_parse']['reason']]+=1
    names=['protocols/decota_spatial_extension_v1.md','vg_tta/decota_spatial_extension_v1.py','scripts/run_decota_spatial_extension_v1.py',
        'methods/CURRENT_METHOD.json','methods/decota_refine_uniform_v1/api.py','methods/decota_refine_uniform_v1/predictor.py',
        'vg_tta/tg_spatial_tta_v1.py','vg_tta/decota_decision_v1.py','vg_tta/decota_cal_spatial_v1.py']
    lock=dict(rows=rows,coverage=dict(coverage),technical_exclusions=excluded,
        removed_watched_from_extension=[r['key'] for r in old if r['split']=='extension' and r['source'] in excluded_sources],
        supplemented=[r['key'] for r in main if r['supplemented']],source_selection_GT=False,
        pins={str(ROOT/f):sha(ROOT/f) for f in names},created=time.time(),old_seal_sha256=sha(F20/'FINAL_SEAL.json'),
        exposure='historically exposed development extension, never untouched confirmation',max_calls=600,max_GPU_seconds=1800,
        temporal_working_readout='original soft; not changed',spatial_time='identical B lowest unlabeled prior loss',
        bootstrap=1000,seed=20260912,epsilon=.001,reference_policy='first accepted old mixed-score candidate0')
    write(OUT/'lock.json',lock)
    print('PREPARED',len(rows),dict(coverage),'replenished',lock['supplemented'],flush=True)


def plan():
    p=read(OUT/'lock.json')
    for f,h in p['pins'].items():assert sha(f)==h,('pin_changed',f)
    return p


def base(row):
    assert sha(row['base_cache'])==row['base_cache_sha256'];x=load(row['base_cache'])
    return (x['predictions']['frozen']['boxes'] if row['supplemented'] else x['boxes']).float()


class ContextView:
    """Same expert forward and OLD NMS score, but explicit target occurrence."""
    def __init__(self,expert,spec):self.model=expert.model;self.processor=expert.processor;self.spec=spec
    def __call__(self,rgb,phrase,entity):
        import torch
        from vg_tta.tg_spatial_tta_v1 import choose_candidate,STOP_WORDS
        from vg_tta.decota_readout_interfaces_v1 import target_tokens
        text=self.spec['context'];inputs=self.processor(images=rgb,text=text,return_tensors='pt').to('cuda')
        tokens=self.processor.tokenizer(text,return_offsets_mapping=True)
        ent=target_tokens(text,tokens['offset_mapping'],self.spec['span'])
        whole=[i for i,(a,b) in enumerate(tokens['offset_mapping']) if b>a and text[a:b].strip().isalpha() and text[a:b] not in STOP_WORDS]
        assert ent and whole and tokens['input_ids']==inputs['input_ids'][0].tolist()
        torch.cuda.synchronize();tick=time.perf_counter();out=self.model(**inputs);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        assert max(ent+whole)<out.logits.shape[-1],'Token span truncation'
        p=out.logits[0].float().sigmoid();target=p[:,ent].mean(-1);mixed=torch.minimum(target,p[:,whole].mean(-1))
        z=choose_candidate(out.pred_boxes[0],mixed)
        z.update(seconds=elapsed,text=text,entity_tokens=ent,phrase_tokens=whole,model_input_shape=list(inputs['pixel_values'].shape),
            all_boxes=out.pred_boxes[0].cpu(),all_phrase_scores=mixed.cpu(),raw_token_logits=out.logits[0].float().cpu(),
            target_all_scores=target.cpu(),target_span=self.spec['span'],offsets=tokens['offset_mapping'],input_ids=tokens['input_ids'])
        return z


def capture():
    import torch
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.decota_decision_v1 import observe
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.exact_frame_decode_audit_v2 import decode
    p=plan();verify_old();configure();start=time.time();calls=0;seconds=0.;receipts=[]
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failure=None
    try:
        config=read(ROOT/'artifacts/decota_refine_v1/lock.json');assert sha(Path(config['expert_snapshot'])/'model.safetensors')==config['expert_sha256']
        expert=SpatialExpert(config['expert_snapshot']);digest=state_digest(expert.model)
        for row in p['rows']:
            key=row['key'];target=dest('capture',key)
            if target.exists():raise RuntimeError('Use a separate explicit resume plan; partial output exists')
            assert time.time()-start<p['max_GPU_seconds'];raw=None;old_probes=[];long=[]
            old=load(row['cache_path']);oldmap={(z['position'],z['text']):z for z in old['expert']}
            if not row['supplemented']:
                sp=checked(file('spatial',key));assert sp['spans']['B']==row['B'] and sp['positions']['B']==row['positions']
                oldmap.update({(z['position'],z['text']):z for z in sp['old_probes']['B']})
            parsed=row['original_parse']
            for j in row['positions']:
                if not parsed['phrase']:continue
                tag=(j,parsed['phrase'].lower().strip()+'.')
                if tag in oldmap:z=oldmap[tag]
                else:
                    if raw is None:raw,ids=decode(row['input']);assert ids==row['frame_ids']
                    z=expert(raw[j],parsed['phrase'],parsed['entity']);z.update(position=j,frame_id=ids[j]);calls+=1;seconds+=z['seconds']
                old_probes.append(z)
            spec=row['context_parse'];reason=spec['reason']
            if spec['eligible']:
                tok=expert.processor.tokenizer(spec['context'])
                if len(tok['input_ids'])>256:reason='token_truncation_keep_S1'
                else:
                    if raw is None:raw,ids=decode(row['input']);assert ids==row['frame_ids']
                    wrapped=ContextView(expert,spec)
                    for j in row['positions']:
                        assert calls<p['max_calls'] and time.time()-start<p['max_GPU_seconds']
                        z=observe(wrapped,raw[j],dict(phrase=spec['context'],entity=spec['entity']));d=z['detection'];calls+=1;seconds+=d['seconds']
                        z.update(position=j,frame_id=ids[j],target_scores=d['target_all_scores'][z['candidate_ids']]);long.append(z)
            assert len(long) in (0,len(row['positions']))
            assert state_digest(expert.model)==digest
            save(target,dict(key=key,old_probes=old_probes,long_probes=long,B=row['B'],positions=row['positions'],
                context_parse=spec,effective_reason=reason,GT_online=False,expert_state_unchanged=True))
            receipts.append(dict(key=key,path=str(target),sha256=sha(target)))
            status(OUT/'progress.json',dict(stage='capture',done=len(receipts),total=len(p['rows']),calls=calls,last=key))
            print('CAPTURE',len(receipts),key,'long',len(long),'total_calls',calls,flush=True)
        write(OUT/'capture_barrier.json',dict(receipts=receipts,created=time.time(),calls=calls,expert_model_seconds=seconds,
            elapsed_seconds=time.time()-start,GT_online=False,expert_sha256=config['expert_sha256'],state_unchanged=True))
    except Exception:failure=traceback.format_exc();raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(seconds=time.time()-start,calls=calls,expert_model_seconds=seconds,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


def predict():
    import torch
    from vg_tta.decota_decision_v1 import fixed_associate
    from vg_tta.decota_cal_spatial_v1 import reconstruct_path
    from methods.decota_refine_uniform_v1.api import reconstruct
    p=plan();barrier=read(OUT/'capture_barrier.json');receipts=[]
    assert [r['key'] for r in p['rows']]==[r['key'] for r in barrier['receipts']]
    for row,receipt in zip(p['rows'],barrier['receipts']):
        assert sha(receipt['path'])==receipt['sha256'];cap=load(receipt['path']);b=base(row);ids=row['frame_ids'];arms={}
        arms['S0']=dict(boxes=b,path=[],support=[],reference=None)
        ps=[{k:z[k] for k in ['position','frame_id','box','score','margin']} for z in cap['old_probes'] if z['accepted']]
        b1,bc=reconstruct(b,ps,ids,'absolute');arms['S1']=dict(boxes=b1,support=[z['position'] for z in ps],pseudo=ps,reconstruction=bc)
        long=cap['long_probes'];gated=[pool_gate(z) for z in long]
        if not long:
            for name in ['S2','S3','S2_ref0','long_old_score']:arms[name]={**arms['S1'],'fallback':'unresolved_context_keep_S1'}
        else:
            path2=[(i,int(np.argmax(z['unary']))) for i,z in enumerate(gated) if z['accepted']]
            path3,ac=fixed_associate(gated,row['weight'])
            ref0=[(i,0 if n==0 else j) for n,(i,j) in enumerate(path2)]
            assert [i for i,j in path2]==[i for i,j in path3]==[i for i,j in ref0]
            for name,path in [('S2',path2),('S3',path3),('S2_ref0',ref0)]:
                boxes,rec=reconstruct_path(b,ids,gated,path)
                arms[name]=dict(boxes=boxes,path=path,support=[gated[i]['position'] for i,j in path],reconstruction=rec,
                    reference=path[0] if path else None,association_audit=ac if name=='S3' else None)
            oldpath=[(i,0) for i,z in enumerate(long) if z['accepted']]
            boxes,rec=reconstruct_path(b,ids,long,oldpath)
            arms['long_old_score']=dict(boxes=boxes,path=oldpath,support=[long[i]['position'] for i,j in oldpath],reconstruction=rec)
        f=dest('predictions',row['key']);save(f,dict(key=row['key'],arms=arms,B=row['B'],GT_online=False,
            capture_sha256=receipt['sha256'],weight=row['weight'],reference_mismatch=bool(gated and any(z['accepted'] for z in gated) and next(z for z in gated if z['accepted'])['admission_winner']!=0)))
        receipts.append(dict(key=row['key'],path=str(f),sha256=sha(f)))
    write(OUT/'prediction_barrier.json',dict(receipts=receipts,created=time.time(),GT_online=False,locked_before_scoring=True))
    print('PREDICTED',len(receipts),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','capture','predict']);args=a.parse_args()
    try:globals()[args.stage]()
    except Exception:write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
