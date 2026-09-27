"""Small RAM-only study. This module never opens target annotations."""
import argparse,copy,fcntl,gc,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.decota_story_common_v1 import *
from methods.decota_v1 import fit,decode
from vg_tta.metrics import interval_from_logits
from vg_tta.native_coverage_calibration_v1 import native_at_length
from vg_tta.posterior_mass_coverage_v1 import select as mass_select

def controls(result,control_config):
    ids=result['frame_ids'];z=result['native_logits'];pred=result['predictions'];boxes=pred['frozen']['boxes']
    pm=mass_select(z,ids,tau=control_config.get('tau',.9))
    for n,ij in [('pm_mass',pm['indices']),('pm_extent_native',native_at_length(z,ids,reference=pm['indices'])['indices']),
                 ('fixed_dev',native_at_length(z,ids,fraction=control_config.get('fraction',.75))['indices'])]:
        pred[n]={'indices':tuple(ij),'boxes':boxes}
    # These placements use precisely the adapted physical length, never GT.
    ij=pred['decota']['indices'];wanted=ids[ij[1]]+1-ids[ij[0]]
    candidates=[(i,j) for i in range(len(ids)) for j in range(i+1,len(ids)) if ids[j]+1-ids[i]==wanted]
    center=(ids[0]+ids[-1]+1)/2
    pred['same_extent_center']={'indices':min(candidates,key=lambda ij:(abs((ids[ij[0]]+ids[ij[1]]+1)/2-center),ij)),'boxes':boxes}
    for seed in [20260909,20260910,20260911]:
        pred[f'same_extent_random{seed}']={'indices':candidates[int(np.random.default_rng(seed).integers(len(candidates)))],'boxes':boxes}
    result['same_extent_candidates']=len(candidates);result['prior_config']=control_config
    return result

def fitted_result(head,inputs,base,backbone,cfg,*,ablations):
    native=base['indices'];z=base['logits'];ids=base['ids'];boxes=base['boxes'];pred={'frozen':{'boxes':boxes,'indices':native}}
    configs={'decota':cfg}
    if ablations:configs.update({'anchor1e4':{**cfg,'gamma':1e-4},'lr0':{**cfg,'lr':0.},**{f'steps{s}':{**cfg,'steps':s} for s in [0,1,3,10,20]}})
    audits={};timing={};state=None
    for name,c in configs.items():
        start=time.perf_counter();private,zs,a=fit(head,inputs,backbone=backbone,**c)
        if backbone=='tastvg':
            from vg_tta.decota_tastvg_episode_v1 import fitted_merge
            extent=fitted_merge(zs,base['views'],ids)
        else:extent=interval_from_logits(zs[0])
        ij=decode(z,extent,ids,native_indices=native)['indices'];pred[name]={'indices':ij,'boxes':boxes};audits[name]=a;timing[name]=time.perf_counter()-start
        if name=='decota':
            pred['coupled_coverage']={'indices':extent,'boxes':boxes};state={k:v.detach().cpu() for k,v in private.state_dict().items()}
        if name in ['lr0','steps0']:
            assert ij==tuple(native)
            assert all(torch.equal(v.cpu(),head.state_dict()[k].cpu()) for k,v in private.state_dict().items())
        del private,zs
    return {'predictions':pred,'native_logits':z.cpu(),'frame_ids':ids,'head_inputs':[h.detach().cpu() for h in inputs],
      'fitted_head_state':state,'native_views':base.get('views'),'audits':audits,'timing':timing,'GT_used':False}

def run(backbone,g,split,condition='clean',limit=0):
    from vg_tta.foreground_runtime import state_digest
    p=verify();rs=cohort(p,g,split,condition);c=next(c for c in p['conditions'] if c['name']==condition);dest=cell(backbone,g,split,condition)
    if (dest/'barrier.json').exists():return
    dev=split=='development';selected={'config':p['groups'][g]['previous_config'] if backbone=='tubedetr' else {'lr':.001,'steps':5,'gamma':0.},'controls':{'fraction':.75,'tau':.9},'sar_margin':.4} if dev else read(config_path(backbone,g))
    if not dev and backbone=='tubedetr':
        override=OUT/'SAR_GATE_RESOLUTION.json'
        if override.exists():selected['sar_margin']=read(override)['groups'][g]['selected_margin']
    cfg=selected['config'];torch.set_num_threads(4);torch.manual_seed(20260909)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX)
    if backbone=='tubedetr' and dev:
        from vg_tta.unanchored_dev_data_v1 import load_source
        spec=read(ROOT/'artifacts/unanchored_fullspan_development_v1/lock.json')['groups'][g]
        head=load_source(spec).cuda().eval();model=None
    elif backbone=='tubedetr':
        from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _lazy_runtime
        from scripts import evaluate_fullspan_scale_shift_external_v1 as external
        rt=_lazy_runtime();rt.add_repo_to_path(ROOT/'external/TubeDETR');model,_=rt.build_model(ROOT/'external/TubeDETR',device='cuda:0',resolution=224,stride=2)
        s=next(v['spec'] for v in p['parents'].values() if v['group']==g)
        assert sha(s['checkpoint'])==s['checkpoint_sha256'];rt.load_official_checkpoint(model,s['checkpoint'])
        model.eval().requires_grad_(False);snapshot=external._snapshot_full(model);head=model.sted_embed
    else:
        from scripts import run_tastvg_span_tta as ta
        from vg_tta.decota_tastvg_episode_v1 import make_batch,episode
        source='hcstvg2' if g=='hc_to_vid' else 'vidstg';ck=p['ta_checkpoints'][source];assert sha(ck['path'])==ck['sha256']
        model,_,_=ta.load_model_on_device(source,ROOT/'artifacts/tastvg_runtime'/('hc-stvg2' if source=='hcstvg2' else 'vidstg'),checkpoint=Path(ck['path']),device='cuda',source_dataset=source)
        model.eval().requires_grad_(False);head=model.temp_embed
    before=state_digest(model if model is not None else head)
    hp=OUT/f'heads/{backbone}_{g}.pt'
    if not hp.exists():save(hp,copy.deepcopy(head).cpu())
    receipts=[];added=0
    for r in rs:
        q=r['input'];idx=int(q['index']);f=dest/'predictions'/f'{idx:06d}.pt';rp=f.with_suffix('.json')
        if rp.exists():
            receipt=read(rp);assert sha(f)==receipt['sha256'];receipts.append(receipt);continue
        start=time.monotonic();torch.cuda.reset_peak_memory_stats();pos=list(range(len(q['frame_ids'])))
        if backbone=='tubedetr' and dev:
            assert sha(r['tube_cache'])==r['tube_cache_sha256'];b=load(r['tube_cache']);assert not b['gt_used']
            z=b['frozen']['pred_sted'];result=fitted_result(head,[b['head_input'].cuda()],{'indices':interval_from_logits(z),'logits':z,'boxes':b['frozen']['pred_boxes'],'ids':q['frame_ids']},backbone,cfg,ablations=False)
            result['raw_input_reuse']={'path':r['tube_cache'],'sha256':r['tube_cache_sha256']}
        else:
            from vg_tta.dense_expansion_data_v1 import decode_raw
            from vg_tta.unanchored_shift_predictor_v1 import predict_episode,_clean_corruption
            from vg_tta.shift_corruptions_v2 import apply_corruption
            raw,ids=decode_raw(q)
            if backbone=='tastvg':
                damaged=_clean_corruption(raw,ids,f'{g}:{idx}',c['seed']) if condition=='clean' else apply_corruption(raw,c['corruption'],c['severity'],query_id=f'{g}:{idx}',seed=c['seed'],frame_ids=ids)
                pos=damaged.metadata['retained_positions'];kept=damaged.metadata['retained_frame_ids'];batch=make_batch(damaged.frames,kept,q,r['subject']['subject'],model)
                baseline=copy.deepcopy(p['baseline_configs']);baseline['sar']['entropy_margin_fraction']=selected['sar_margin']
                result=episode(model,batch,cfg,baseline,baselines=True,ablations=(not dev and condition=='clean'))
                result['realized_corruption']=damaged.metadata;del batch,damaged
            else:
                b=predict_episode(model,raw,ids,q['caption'],f'{g}:{idx}',corruption=c['corruption'],severity=c['severity'],seed=c['seed'],selected_config=cfg,
                   source_snapshot=snapshot,device='cuda:0',baselines=True,sar_entropy_margin_fraction=selected['sar_margin'])
                n=b['native_predictions'];pos=b['frame_mapping']['retained_positions'];kept=b['frame_mapping']['retained_frame_ids'];z=n['frozen']['pred_sted']
                result=fitted_result(head,[b['head_input'].cuda()],{'indices':interval_from_logits(z),'logits':z,'boxes':n['frozen']['pred_boxes'],'ids':kept},backbone,cfg,ablations=condition=='clean')
                assert result['predictions']['coupled_coverage']['indices']==interval_from_logits(n['ours']['pred_sted'])
                result['baseline_audits']=b['audits']['external_methods'];result['realized_corruption']=b['realized_corruption']
                for m in ['tent','memo','sar']:result['predictions'][m]={'indices':interval_from_logits(n[m]['pred_sted']),'boxes':n[m]['pred_boxes']}
            del raw
        result=controls(result,selected['controls'])
        from scripts.run_decota_matrix_v1 import lift
        result=lift(result,pos,q['frame_ids'])
        result.update(backbone=backbone,group=g,index=idx,source=q['source'],split=split,condition=c,config=cfg,
            lock_sha256=sha(OUT/'lock.json'),peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,episode_seconds=time.monotonic()-start)
        save(f,result);receipt={'path':str(f),'sha256':sha(f),'index':idx,'source':q['source'],'GT_used':False};write(rp,receipt);receipts.append(receipt)
        status(OUT/'cell_progress.json',{'backbone':backbone,'group':g,'split':split,'condition':condition,'completed':len(receipts),'total':len(rs),'seconds':result['episode_seconds'],'time':time.time()})
        print('PREDICT',backbone,g,split,condition,len(receipts),len(rs),round(result['episode_seconds'],2),flush=True);del result;gc.collect();added+=1
        if limit and added>=limit:break
    assert before==state_digest(model if model is not None else head)
    if len(receipts)==len(rs):write(dest/'barrier.json',{'queries':len(rs),'receipts':receipts,'state_unchanged':True,'GT_used':False,'lock_sha256':sha(OUT/'lock.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--backbone',choices=BACKBONES);ap.add_argument('--group',choices=GROUPS);ap.add_argument('--split',choices=['development','confirmation']);ap.add_argument('--condition',default='clean');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:run(a.backbone,a.group,a.split,a.condition,a.limit)
