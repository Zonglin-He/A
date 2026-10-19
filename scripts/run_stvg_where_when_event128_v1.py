"""Finite HC128 semantic extraction then sealed readout. No scoring imports."""
from pathlib import Path
import sys,time,os,json,collections,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_where_when_event128_v1';P=R/'results/stvg_error_state/2026-10-19/where_when_event128'
PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FINAL=R/'artifacts/stvg_referdino_finalize_v1'

def prepare():
    assert not D.exists();D.mkdir();P.mkdir(parents=True)
    old=read(PRE/'PLAN_LOCK.json');cfg=read(R/'methods/stvg_referdino_opd_v1/config.json')
    assert cfg['tau']==.2 and cfg['seeds']==[41,42,43]
    records=[];raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
    for r in old['records']:
        a=r['alias'];row=raw[r['query_ordinal']];subject=R/f"artifacts/decota_paper_experiments_v1/hc2/subjects/{r['query_ordinal']:05}.json"
        records.append(dict(r,subject_path=str(subject.relative_to(R)),subject_sha256=sha(subject),teacher0_sha256=sha(PRE/'teacher0'/f'{a}.pt'),spatial_input_sha256=sha(PRE/'inputs'/f'{a}.pt'),live_sha256=sha(FINAL/'Live/predictions'/f'{a}.pt'),caption_sha256=__import__('hashlib').sha256(row['input']['caption'].encode()).hexdigest(),frame_count=row['input']['frame_count']))
    code=['vg_tta/stvg_where_when_event128_v1.py','scripts/run_stvg_where_when_event128_v1.py','scripts/score_stvg_where_when_event128_v1.py','protocols/stvg_where_when_event128_v1.md','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/experts.py','vg_tta/stvg_referdino_condition_hc128_v1.py','external/ReferDINO/models/GroundingDINO/groundingdino.py','external/ReferDINO/models/GroundingDINO/utils.py']
    plan=dict(status='locked_before_new_inference_and_GT',records=records,count=128,seeds=[41,42,43],created_epoch=time.time(),deadline_epoch=time.time()+10800,GPU_seconds_cap=900,pins={f:sha(R/f)for f in code},frozen_spatial_config=cfg,cohort='all existing RC128, exposed development, not confirmation',GT_read=False,configuration=dict(teacher='frozen ryt_mevis_swinb',frames=8,new_Source=0,new_DINO_T=0,new_spatial_updates=0,new_experts=0,new_teacher_video_queries=128,new_teacher_image_frames=1024,policy_updates=0,association='global mean8 IoU argmax, lowest query tie, retain zero overlap diagnostic',event_rule='all lexical VERB seeds or copular predicate; nonsubject subtrees, word-balanced subword-mean logits',interval_rule='Voronoi-cell-edge contiguous intervals plus original MBR/full video, half-open physical frames; duration-weighted inside-minus-outside mean; fullspan contrast unknown; no event/flat response returns MBR; tie nearest MBR then earlier endpoints',readouts=['NativeConfidence','EventNative','EventE','EventLive41','EventLive42','EventLive43'],new_temporal_OPD=False,parameter_search=False))
    put(D/'PLAN_LOCK.json',plan);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in plan.items()if k not in ['records','pins']});put(D/'STATUS.json',dict(stage='prepared',done=0,total=128),True)

def verify():
    from main_method.runtime import resource_guard
    p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert time.time()<p['deadline_epoch'];resource_guard();return p

def extract():
    import torch,numpy as np,hashlib
    from methods.stvg_referdino_opd_v1.runtime import setup,lease
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    from vg_tta.stvg_referdino_condition_hc128_v1 import ConditionalRefer
    from methods.decota_final_simplified_v1.tensors import state_hash
    setup();torch.manual_seed(42);np.random.seed(42);plan=verify();lock=lease();start=time.perf_counter();files=[];calls=0
    used=sum(read(f)['GPU_seconds']for f in (D/'cost').glob('*.json'))
    try:
        model=ConditionalRefer();mh=state_hash(model.model.state_dict());load_seconds=time.perf_counter()-start
        for r in plan['records']:
            a=r['alias'];f=D/'semantics'/f'{a}.pt'
            if f.exists():files.append(read(f.with_suffix('.receipt.json')));continue
            verify();assert time.perf_counter()-start+used+3<plan['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
            assert sha(PRE/'teacher0'/f'{a}.pt')==r['teacher0_sha256'];old=load(PRE/'teacher0'/f'{a}.pt');row=read_row('hc2',r['query_ordinal']);t=time.perf_counter();frames,ids,_=frames_for('hc2',row,collections.OrderedDict());rgb=frames[r['positions8']];decode_s=time.perf_counter()-t
            assert [hashlib.sha256(x.tobytes()).hexdigest()for x in rgb]==old['rgb_sha256']
            cap={};shapes=[]
            def hook(module,args,result):
                x,td=args;shapes.append(list(result.shape));cap.update(premax=result.detach().cpu().clone(),mask=td['text_token_mask'].detach().cpu().clone())
            h=model.model.class_embed[-1].register_forward_hook(hook)
            try:new=model.run(rgb,row['input']['caption']);calls+=1
            finally:h.remove()
            trace=new['trace'];assert torch.equal(cap['premax'].max(-1,keepdim=True).values,trace['all_logits'])
            assert torch.equal(trace['all_logits'],old['native']['trace']['all_logits']) and torch.equal(trace['all_boxes'],old['native']['trace']['all_boxes'])
            assert torch.equal(new['boxes'],old['native']['boxes']) and new['selected_query']==old['native']['selected_query']
            text=' '.join(row['input']['caption'].lower().split());text=text if text.endswith('.')else text+'.'
            tk=model.model.tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True)
            L=min(tk['input_ids'].shape[-1],model.model.max_text_len);assert cap['mask'].shape[-1]==L and torch.equal(cap['mask'],tk['attention_mask'][:,:L].bool().expand(8,-1))
            payload=dict(alias=a,premax=cap['premax'],text_mask=cap['mask'],token_ids=tk['input_ids'][0,:L].cpu(),offsets=tk['offset_mapping'][0,:L].tolist(),all_boxes=trace['all_boxes'],all_logits=trace['all_logits'],selected_query=new['selected_query'],tracked_lineages=trace['tracked_lineages'],matching_permutations=trace['matching_permutations'],hook_shapes=shapes,normalized_caption_sha256=hashlib.sha256(text.encode()).hexdigest(),frame_ids8=[ids[i]for i in r['positions8']],legacy_bitexact=True,token_max_bitexact=True,cost=dict(new['cost'],decode_seconds=decode_s),GT_read=False)
            files.append(save(f,payload));put(D/'STATUS.json',dict(stage='extract',done=len(files),total=128,GT_read=False,elapsed_seconds=time.perf_counter()-start),True);print('semantic',a,round(time.perf_counter()-start,2),flush=True)
            del frames,rgb,new,old,payload,cap;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.model.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in model.model.parameters())
        put(D/'SEMANTIC_BARRIER.json',dict(count=128,files=files,weights_unchanged=True,all_legacy_native_outputs_bitexact=True,GT_read=False))
    finally:
        put(D/'cost'/f'extract_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-start+3,new_teacher_video_queries=calls,new_teacher_image_frames=8*calls,model_load_seconds=locals().get('load_seconds'),complete=len(files)==128));lock.close()

def predict():
    import numpy as np,torch
    from vg_tta.stvg_where_when_event128_v1 import event_words,word_tokens,semantic_curve,associate,candidates,choose
    p=verify();bar=read(D/'SEMANTIC_BARRIER.json');assert bar['count']==128;files=[]
    raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
    for r in p['records']:
        a=r['alias'];sem=load(D/'semantics'/f'{a}.pt');assert sha(PRE/'inputs'/f'{a}.pt')==r['spatial_input_sha256'] and sha(FINAL/'Live/predictions'/f'{a}.pt')==r['live_sha256']
        z=load(PRE/'inputs'/f'{a}.pt');s=load(FINAL/'Live/predictions'/f'{a}.pt');assert sha(R/r['subject_path'])==r['subject_sha256'];sub=read(R/r['subject_path']);caption=raw[r['query_ordinal']]['input']['caption']
        tokens=event_words(sub['parses']['context']['tokens']);groups,reason=word_tokens(caption,tokens,sem['offsets']);assert not groups or all(sem['text_mask'][:,g].all()for g in groups)
        edges,cs=candidates(z['frame_ids'],r['positions8'],r['frame_count'],z['interval']);boxes=sem['all_boxes'].double().numpy();native=sem['selected_query'];curves={'NativeConfidence':sem['all_logits'][:,native,0].sigmoid().double().numpy(),'EventNative':semantic_curve(sem['premax'],groups,native)};assoc={};j,e=associate(z['E'][r['positions8']],boxes);assoc['E']=dict(query=j,**e);curves['EventE']=semantic_curve(sem['premax'],groups,j)
        trajectories={'E':np.asarray(z['E'])}
        for seed in p['seeds']:
            b=s['outputs'][f'T20_Live_{seed}'];trajectories[f'Live{seed}']=np.asarray(b);j,e=associate(b[r['positions8']],boxes);assoc[f'Live{seed}']=dict(query=j,**e);curves[f'EventLive{seed}']=semantic_curve(sem['premax'],groups,j)
        readouts={}
        for name,curve in curves.items():
            iv,scores,why=choose(curve,edges,cs,z['interval']);readouts[name]=dict(interval=iv,curve=None if curve is None else curve.tolist(),candidate_scores=scores,reason=why,changed=iv!=list(z['interval']))
        files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],positions8=r['positions8'],original_interval=z['interval'],physical_frame_ids8=sem['frame_ids8'],frame_count=r['frame_count'],edges=edges.tolist(),candidates=cs,association=assoc,native_query=native,event_group_token_ids=groups,event_word_count=len(tokens),event_status=reason,readouts=readouts,trajectories=trajectories,GT_read=False)))
        del s,z,sem;gc.collect()
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,all_outputs_before_new_GT=True,semantic_barrier_sha256=sha(D/'SEMANTIC_BARRIER.json'),plan_sha256=sha(D/'PLAN_LOCK.json'),GT_read=False));put(D/'STATUS.json',dict(stage='predictions_sealed',done=128,total=128,GT_read=False),True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','extract','predict']);a=ap.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:
        D.mkdir(exist_ok=True,parents=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,error=str(e),type=type(e).__name__,partial_outputs_preserved=True));raise
