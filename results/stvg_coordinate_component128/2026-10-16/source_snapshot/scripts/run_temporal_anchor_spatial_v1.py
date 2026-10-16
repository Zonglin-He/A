#!/usr/bin/env python3
"""Restore the positive domain-level anchor and isolate spatial-head development."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_phase2_support_query import dataset_annotation_rows, load_fixed_split
from scripts.run_feasibility import load_full_video, prediction_record
from vg_tta.metrics import interval_from_logits, compute_stvg_metrics
from vg_tta.phase2 import deterministic_views, source_cluster
from vg_tta.tta import EpisodicHeadAdapter
from vg_tta.tubedetr_runtime import build_model, load_official_checkpoint, dataset_args, forward_video

OUT=ROOT/'artifacts/temporal_anchor_spatial_v1'
PARENT=ROOT/'artifacts/decoder_v1/phase2_natural_vidstg_to_hcstvg2/manifest.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return value


def predict_capture(model,video,caption):
    captured=[]
    handle=model.bbox_embed.register_forward_pre_hook(lambda _m,inp:captured.append(inp[0].detach()))
    try:
        with torch.no_grad():
            outputs=forward_video(model,video,caption,repo=ROOT/'external/TubeDETR',stride=2,device='cuda')
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError('bbox head input not captured exactly once')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        reapplied=model.bbox_embed(captured[0]).sigmoid()[-1]
    if not torch.equal(reapplied,outputs['pred_boxes']):
        raise RuntimeError('Cached native bbox head is not exact')
    return cpu({k:outputs[k] for k in ('pred_boxes','pred_sted')}),cpu(captured[0])


def evaluation_record(output,targets,video_target,annotation,index,method,seed):
    result=prediction_record(output,targets,video_target,annotation,sample_index=index,
        condition='clean',method=method,runtime_sec=0,peak_vram_gb=0)
    result.update(source=source_cluster(annotation['video_path']),support_seed=seed,
        support_size=0 if method=='frozen' else 64,evidence_role='previously_exposed_development')
    return result


def load_runtime():
    torch.set_num_threads(4);torch.manual_seed(20260906);np.random.seed(20260906)
    torch.backends.cudnn.benchmark=False
    model,args=build_model(ROOT/'external/TubeDETR',device='cuda',resolution=224,stride=2)
    parent=json.loads(PARENT.read_text())
    checkpoint=ROOT/parent['args']['checkpoint']
    if sha(checkpoint)!=parent['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    load=load_official_checkpoint(model,checkpoint)
    if load['missing_keys'] or load['unexpected_keys']:raise RuntimeError(load)
    model.eval()
    from datasets import build_dataset
    data=build_dataset('hcstvg',image_set='val',args=dataset_args(args,
        ROOT/'data/hcstvg2_confirm512',ROOT/'data/hcstvg2_confirm512/annotations'))
    annotations=dataset_annotation_rows(data)
    split,source=load_fixed_split(PARENT,annotations,support_sizes=[64],
        support_seeds=[41001,41002,41003],dataset='hcstvg',
        annotation_file=ROOT/'data/hcstvg2_confirm512/annotations/valv2_proc.json',
        video_root=ROOT/'data/hcstvg2_confirm512')
    return model,data,annotations,split,source,load


def make_lock(split,source,load):
    codepaths=[Path(__file__),ROOT/'protocols/temporal_anchor_spatial_v1.json',
        ROOT/'vg_tta/tta.py',ROOT/'vg_tta/metrics.py',ROOT/'vg_tta/phase2.py',
        ROOT/'vg_tta/augmentations.py',ROOT/'vg_tta/tubedetr_runtime.py',
        ROOT/'scripts/run_phase2_support_query.py',ROOT/'scripts/run_feasibility.py']
    codepaths+=list((ROOT/'external/TubeDETR/models').rglob('*.py'))
    codepaths+=list((ROOT/'external/TubeDETR/datasets').rglob('*.py'))
    return {'version':'temporal_anchor_spatial_v1','protocol':json.loads((ROOT/'protocols/temporal_anchor_spatial_v1.json').read_text()),
        'parent_sha256':sha(PARENT),'code_sha256':{str(p):sha(p) for p in codepaths},
        'split':split,'split_source':source,'checkpoint_load':load}


def anchor_and_cache(args):
    model,data,annotations,split,source,load=load_runtime()
    lock=make_lock(split,source,load)
    path=OUT/'lock.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise RuntimeError('Existing locked code/inputs differ')
    write(path,lock)
    adapter=EpisodicHeadAdapter(model,scope='heads_ln_projection',learning_rate=1e-4,
        weight_decay=0,optimizer_eps=1e-4,minimum_loss_for_update=None)
    query_indices=split['query_indices'][:args.query_limit] if args.query_limit else split['query_indices']
    # Frozen evaluation uses no support information.
    if not args.skip_frozen:
        for position,index in enumerate(query_indices):
            dest=OUT/'frozen'/f'query_{index:05d}.json'
            if dest.exists():continue
            video,targets,vt=load_full_video(data[index])
            output,hidden=predict_capture(model,video,vt['caption'])
            rec=evaluation_record(output,targets,vt,annotations[index],index,'frozen',0)
            full=compute_stvg_metrics(output['pred_boxes'],targets,(0,len(targets)-1),tuple(vt['inter_idx']),
                frame_ids=vt['frames_id'],gt_frame_interval=(annotations[index]['tube_start_frame'],annotations[index]['tube_end_frame']))
            write(dest,{'lock_sha256':sha(path),'record':rec,'full_span_metrics':full})
            if (position+1)%20==0 or position+1==len(query_indices):print(f'[frozen] {position+1}/{len(query_indices)}',flush=True)
    for seed in args.seeds:
        adapter.reset()
        dest=OUT/f'seed_{seed}'
        checkpoint=dest/'temporal_adapted_parameters.pt'
        support_indices=split['support_plans'][str(seed)]['ordered_indices'][:64]
        if args.support_limit:support_indices=support_indices[:args.support_limit]
        if checkpoint.exists():
            obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if obj['lock_sha256']!=sha(path) or obj['support_indices']!=support_indices:raise RuntimeError('Checkpoint resume provenance changed')
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if name in obj['parameters']:param.copy_(obj['parameters'][name].to(param))
        else:
            updates=[]
            for position,index in enumerate(support_indices):
                video,_targets,vt=load_full_video(data[index])
                output=forward_video(model,video,vt['caption'],repo=ROOT/'external/TubeDETR',stride=2,device='cuda')
                loss=adapter.step_temporal_entropy(output)
                bbox_gradients={n:p.grad is not None for n,p in model.named_parameters() if n.startswith('bbox_embed.')}
                if any(bbox_gradients.values()):raise RuntimeError('Unexpected bbox direct entropy gradient')
                updates.append({'index':index,'source':source_cluster(annotations[index]['video_path']),
                    'loss':loss,'bbox_gradient_present':bbox_gradients,'gt_used':False})
                del output,video
                if (position+1)%16==0:print(f'[temporal seed={seed}] {position+1}/{len(support_indices)}',flush=True)
            parameters={n:cpu(p) for n,p in model.named_parameters() if n in adapter.parameter_names}
            bbox_delta=max(float((parameters[n]-adapter._initial_state[n].cpu()).abs().max()) for n in parameters if n.startswith('bbox_embed.'))
            if bbox_delta!=0:raise RuntimeError('Entropy mutated bbox head')
            dest.mkdir(parents=True,exist_ok=True)
            torch.save({'parameters':parameters,'lock_sha256':sha(path),'support_indices':support_indices},checkpoint)
            write(dest/'temporal_updates.json',{'updates':updates,'bbox_parameter_max_abs_delta':bbox_delta,'gt_used':False})
        model.eval().requires_grad_(False)
        torch.save(cpu(model.bbox_embed.state_dict()),dest/'bbox_initial.pt')
        # All teacher/cache construction uses a fixed temporal-adapted model.
        for role,indices in [('support',support_indices),('query',query_indices)]:
            for position,index in enumerate(indices):
                cache_path=dest/(role+'_cache')/f'clip_{index:05d}.pt'
                if cache_path.exists():continue
                video,targets,vt=load_full_video(data[index])
                outputs,hidden=predict_capture(model,video,vt['caption'])
                predictions=[outputs]
                for view in deterministic_views(video,seed=20260906+index*10,count=3)[1:]:
                    with torch.no_grad():
                        out=forward_video(model,view,vt['caption'],repo=ROOT/'external/TubeDETR',stride=2,device='cuda')
                    predictions.append(cpu({k:out[k] for k in ('pred_boxes','pred_sted')}))
                    del out,view
                teacher=torch.stack([o['pred_boxes'].float() for o in predictions]).median(0).values
                start,end=interval_from_logits(outputs['pred_sted'])
                weights=torch.zeros(len(teacher));weights[start:end+1]=1
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                # The training record is explicitly label-free and isolated from evaluation/oracle targets.
                cache={'head_input':hidden,'teacher_boxes':teacher,'weights':weights,
                    'anchor_prediction':outputs,'identity_teacher_boxes':outputs['pred_boxes'],
                    'index':index,'source':source_cluster(annotations[index]['video_path']),
                    'lock_sha256':sha(path)}
                torch.save(cache,cache_path)
                eval_path=dest/(role+'_evaluation_only')/f'clip_{index:05d}.pt'
                eval_path.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'targets':cpu([{'boxes':t.get('boxes',torch.empty(0,4))} for t in targets]),
                    'video_target':{k:vt[k] for k in ('caption','video_id','frames_id','inter_idx')},
                    'annotation':annotations[index]},eval_path)
                if role=='query':
                    rec=evaluation_record(outputs,targets,vt,annotations[index],index,'temporal_anchor',seed)
                    ensembled={**outputs,'pred_boxes':teacher}
                    aug=evaluation_record(ensembled,targets,vt,annotations[index],index,'spatial_ensemble_no_update',seed)
                    write(dest/'query_records'/f'query_{index:05d}.json',{'records':[rec,aug],'lock_sha256':sha(path)})
                if (position+1)%16==0 or position+1==len(indices):print(f'[cache seed={seed} {role}] {position+1}/{len(indices)}',flush=True)
                del video,hidden,cache,predictions
        write(dest/'anchor_cache_complete.json',{'lock_sha256':sha(path),'query_count':len(query_indices),
            'support_count':len(support_indices),'checkpoint_sha256':sha(checkpoint),
            'bbox_reapplication_exact':True,'limited_smoke':bool(args.query_limit or args.support_limit)})
    adapter.reset()


def spatial(args):
    from vg_tta.spatial_support import fit_spatial_head, replay_box_head
    model,data,annotations,split,source,load=load_runtime()
    lock=json.loads((OUT/'lock.json').read_text())
    if make_lock(split,source,load)!=lock:raise RuntimeError('Locked anchor code/inputs changed')
    for seed in args.seeds:
        folder=OUT/f'seed_{seed}'
        completion=json.loads((folder/'anchor_cache_complete.json').read_text())
        if completion['limited_smoke'] and not args.allow_smoke:raise RuntimeError('Cannot call incomplete anchor formal spatial experiment')
        spatial_lock={'anchor_lock':sha(OUT/'lock.json'),'module_sha256':sha(ROOT/'vg_tta/spatial_support.py')}
        if (folder/'spatial_lock.json').exists() and json.loads((folder/'spatial_lock.json').read_text())!=spatial_lock:
            raise RuntimeError('Spatial module changed')
        write(folder/'spatial_lock.json',spatial_lock)
        support_indices=split['support_plans'][str(seed)]['ordered_indices'][:completion['support_count']]
        query_indices=split['query_indices'][:completion['query_count']]
        caches={i:torch.load(folder/'support_cache'/f'clip_{i:05d}.pt',map_location='cpu',weights_only=False) for i in support_indices}
        entries=[{k:caches[i][k] for k in ('head_input','teacher_boxes','weights')} for i in support_indices]
        oracle=[]
        for i in support_indices:
            gt=torch.load(folder/'support_evaluation_only'/f'clip_{i:05d}.pt',map_location='cpu',weights_only=False)
            boxes=torch.zeros_like(caches[i]['teacher_boxes']);mask=torch.zeros(len(boxes))
            for t,target in enumerate(gt['targets']):
                if target['boxes'].numel():boxes[t]=target['boxes'][0];mask[t]=1
            oracle.append({'head_input':caches[i]['head_input'],'teacher_boxes':boxes,'weights':mask})
        initial=torch.load(folder/'bbox_initial.pt',map_location='cpu',weights_only=False)
        variants=[('spatial_lr1e-5',1e-5,entries,False),('spatial_lr1e-4',1e-4,entries,False),
            ('spatial_lr1e-3',1e-3,entries,False),('spatial_lr0',0.,entries,False),
            ('identity_teacher',1e-4,[{**r,'teacher_boxes':caches[i]['identity_teacher_boxes']} for r,i in zip(entries,support_indices)],False),
            ('gt_spatial_oracle_lr1e-4',1e-4,oracle,True),('gt_spatial_oracle_lr1e-3',1e-3,oracle,True)]
        for name,lr,records,gt_used in variants:
            report_path=folder/'spatial_results'/f'{name}.json'
            if report_path.exists():continue
            head=copy.deepcopy(model.bbox_embed).eval().requires_grad_(True)
            head.load_state_dict(initial)
            began=time.perf_counter()
            diagnostic=fit_spatial_head(head,records,lr=lr,anchor_gamma=1e-4,optimizer_eps=1e-4)
            results=[]
            for i in query_indices:
                cached=torch.load(folder/'query_cache'/f'clip_{i:05d}.pt',map_location='cpu',weights_only=False)
                target=torch.load(folder/'query_evaluation_only'/f'clip_{i:05d}.pt',map_location='cpu',weights_only=False)
                with torch.no_grad():boxes=replay_box_head(head,cached['head_input'],device='cuda')
                output={**cached['anchor_prediction'],'pred_boxes':boxes.detach().cpu()}
                if lr==0 and not torch.equal(output['pred_boxes'],cached['anchor_prediction']['pred_boxes']):raise RuntimeError('lr0 bbox not exact')
                if not torch.equal(output['pred_sted'],cached['anchor_prediction']['pred_sted']):raise RuntimeError('Temporal changed')
                rec=evaluation_record(output,target['targets'],target['video_target'],target['annotation'],i,name,seed)
                rec.update(gt_used=gt_used,temporal_anchor_exact=True)
                results.append(rec)
            report_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(cpu(head.state_dict()),report_path.with_suffix('.pt'))
            write(report_path,{'records':results,'diagnostics':diagnostic,'gt_used':gt_used,
                'spatial_lock':spatial_lock,'head_sha256':sha(report_path.with_suffix('.pt')),
                'seconds':time.perf_counter()-began,'query_updates':0})
            print(f'[spatial seed={seed}] {name} done {time.perf_counter()-began:.2f}s',flush=True)
            del head
        write(folder/'spatial_complete.json',{'spatial_lock':spatial_lock,'variants':len(variants),
            'query_count':len(query_indices),'support_count':len(support_indices),'temporal_anchor_exact':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['anchor','spatial'])
    p.add_argument('--seeds',nargs='+',type=int,default=[41001,41002,41003])
    p.add_argument('--query-limit',type=int);p.add_argument('--support-limit',type=int)
    p.add_argument('--skip-frozen',action='store_true');p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    global OUT
    if args.output:OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    (anchor_and_cache if args.mode=='anchor' else spatial)(args)


if __name__=='__main__':main()
