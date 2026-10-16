"""Versioned, no-retuning sparse correction study; prediction path is GT-free."""
import argparse,gc,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/decota_refine_v1';PREV=ROOT/'artifacts/decota_s_v1';STORY=ROOT/'artifacts/decota_story_v1'


def prepare():
    from vg_tta.foreground_runtime import QuerySubjectParser
    from vg_tta.tg_spatial_tta_v1 import visual_query
    old=read(PREV/'lock.json');s=read(STORY/'lock.json');parser=QuerySubjectParser(ROOT/'.cache/stanza')
    excluded=[r['input'] for part in ['development','evaluation'] for rows in old[part].values() for r in rows]
    source_ex={r['source'] for r in excluded};hash_ex={r['video_sha256'] for r in excluded}
    p=dict(version='decota_refine_v1',created_unix=time.time(),seed=20260910,
        parent_lock_sha256=sha(PREV/'lock.json'),parent_report_sha256=sha(PREV/'REPORT.md'),
        promotion='Residual interpolation promoted AFTER observing the old 128-source comparison. Never retroactively pre-registered.',
        main='DeCoTA temporal parameter TTA + nonparametric residual interpolation',
        no_new_hyperparameter_selection=True,spatial_gradient_steps=0,expert_frames=8,
        expert_snapshot=old['expert_snapshot'],expert_sha256=old['expert_sha256'],ta_checkpoints=old['checkpoints'],
        temporal_configs={b:{g:read(STORY/f'tuning/{b}/{g}/selected.json')['config'] for g in old['evaluation']} for b in ['tastvg','tubedetr']},
        tube_checkpoints={},labels=old['evaluation_label_specs'],history=old['evaluation'],new={},
        methods=['frozen','decota','direct','absolute','residual','gsi_official','gsi_matched'],
        gsi=dict(repo='https://github.com/dyhBUPT/StrongSORT',commit='ee995076da5083e28d0da1f885297df62705ebd7',
            tau=10,interval=20,units='physical original frame IDs',coordinate='normalized TLWH',
            api_compatibility='reshape sklearn single-target prediction to N x 1; values/defaults unchanged',
            matched_variant='GaussianSmooth on absolute-interpolated sampled anchor hull; gap cutoff removed, explicitly an adaptation'),
        no_anchor='native unchanged',one_anchor='only that frame',outside_hull='native unchanged',
        correction_coordinates='normalized cxcywh',interpolation_time='physical original frame IDs',
        old_query_tta='Historical TA-STVG comparison only; no new spatial-gradient search/branch on new sources or TubeDETR')
    audit={}
    for g in old['evaluation']:
        candidates={}
        for parent,v in s['parents'].items():
            if v['group']!=g:continue
            p['tube_checkpoints'][g]={k:v['spec'][k] for k in ['checkpoint','checkpoint_sha256']}
            for q in v['spec']['queries']:
                if q['source'] in source_ex or q['video_sha256'] in hash_ex or not Path(q['video_path']).is_file():continue
                candidates.setdefault(q['source'],[]).append((parent,q))
        chosen_sources=sorted(candidates,key=lambda src:hashlib.sha256(('decota_refine_v1:'+str(src)).encode()).hexdigest())[:64]
        assert len(chosen_sources)==64
        rows=[]
        for ordinal,source in enumerate(chosen_sources):
            parent,q=min(candidates[source],key=lambda v:hashlib.sha256((v[0]+':'+str(v[1]['index'])+':'+v[1]['caption']).encode()).hexdigest())
            rows.append(dict(ordinal=ordinal,input=q,parent=parent,student_subject=parser(q['caption']),visual_query=visual_query(parser,q['caption'])))
        assert len({r['input']['video_sha256'] for r in rows})==64
        p['new'][g]=rows
        audit[g]=dict(available_sources=len(candidates),selected_sources=64,source_overlap=0,media_hash_overlap=0,
            independent_of_declared_spatial_selection_rosters=True,historically_untouched=False,
            note='Excludes 64 development and 128 old evaluation sources globally. Other historical experiments may have used these videos. No global never-seen claim.')
        for r in p['history'][g]:
            r['cache_path']=str(PREV/'evaluation'/g/f"{r['input']['index']:06d}.pt");r['cache_sha256']=sha(r['cache_path'])
    files=['methods/decota_refine_v1/__init__.py','methods/decota_refine_v1/api.py','methods/decota_refine_v1/predictor.py',
        'scripts/run_decota_refine_v1.py','methods/decota_s_v1/api.py','methods/decota_v1/api.py','methods/decota_v1/_fullspan.py',
        'vg_tta/tg_spatial_tta_v1.py','vg_tta/decota_tastvg_episode_v1.py','vg_tta/unanchored_shift_predictor_v1.py',
        'external/StrongSORT/GSI.py','vg_tta/metrics.py','vg_tta/st_component_diagnostics_v1.py']
    p['pins']={f:sha(ROOT/f) for f in files};write(OUT/'lock.json',p);write(OUT/'DATA_AUDIT.json',audit)
    print('LOCKED: historical 128; 128 other sources x 2 backbones; no parameter search',flush=True)


def plan():
    p=read(OUT/'lock.json')
    from scripts.decota_refine_protocol_v1 import verify_code
    verify_code(p)
    return p


def student(p,backbone,g):
    if backbone=='tastvg':
        from scripts import run_tastvg_span_tta as ta
        source='hcstvg2' if g=='hc_to_vid' else 'vidstg';ck=p['ta_checkpoints'][source]
        assert sha(ck['path'])==ck['sha256']
        m,*_=ta.load_model_on_device(source,ROOT/'artifacts/tastvg_runtime'/('hc-stvg2' if source=='hcstvg2' else source),
            checkpoint=Path(ck['path']),device='cuda',source_dataset=source)
    else:
        from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _lazy_runtime
        rt=_lazy_runtime();rt.add_repo_to_path(ROOT/'external/TubeDETR')
        m,_=rt.build_model(ROOT/'external/TubeDETR',device='cuda:0',resolution=224,stride=2)
        ck=p['tube_checkpoints'][g];assert sha(ck['checkpoint'])==ck['checkpoint_sha256'];rt.load_official_checkpoint(m,ck['checkpoint'])
    return m.eval().requires_grad_(False)


def configure():
    import torch,numpy as np
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def smoke():
    import torch
    from methods.decota_refine_v1.predictor import DeCoTARefinePredictor
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.dense_expansion_data_v1 import decode_raw
    configure();p=plan();expert=SpatialExpert(p['expert_snapshot']);results={}
    for b in ['tastvg','tubedetr']:
        for g in p['history']:
            row=p['history'][g][0];q=row['input'];m=student(p,b,g)
            pred=DeCoTARefinePredictor(m,expert,None,backbone=b,temporal_config=p['temporal_configs'][b][g])
            frames,ids=decode_raw(q);r=pred.predict(frames,ids,q,subject=row['student_subject']['subject'],parsed_visual_query=row['visual_query'])
            br=read(STORY/f'runs/{b}/{g}/confirmation/clean/barrier.json');receipt=next(z for z in br['receipts'] if z['index']==q['index'])
            assert sha(receipt['path'])==receipt['sha256'];old=load(receipt['path'])
            assert torch.equal(r['runtime']['boxes'],old['predictions']['frozen']['boxes'])
            assert r['indices']==list(old['predictions']['decota']['indices'])
            if b=='tastvg':
                cached=load(row['cache_path']);assert torch.equal(r['boxes'],cached['predictions']['interpolation']['boxes'])
                assert r['keyframes']==cached['keyframe_sets']['main']
            results[b+'_'+g]=dict(index=q['index'],native_exact=True,temporal_exact=True,spatial_backward_calls=0)
            print('SMOKE',b,g,results[b+'_'+g],flush=True)
            del frames,r,pred,m;gc.collect();torch.cuda.empty_cache()
    write(OUT/'SMOKE_AUDIT.json',results)


def history():
    import torch
    from threadpoolctl import threadpool_limits
    from methods.decota_refine_v1.api import spatial_variants
    p=plan();configure()
    with threadpool_limits(limits=1):
        for g,rows in p['history'].items():
            for j,row in enumerate(rows):
                f=OUT/'history'/'tastvg'/g/f'{j:03d}.pt'
                if f.exists():continue
                assert sha(row['cache_path'])==row['cache_sha256'];x=load(row['cache_path']);base=x['predictions']['decota']['boxes'];pseudo=x['supervision']['main']
                variants=spatial_variants(base,pseudo,x['frame_ids'])
                assert torch.equal(variants['residual'][0],x['predictions']['interpolation']['boxes'])
                assert torch.equal(variants['direct'][0],x['predictions']['direct_sparse']['boxes'])
                preds={k:x['predictions'][k] for k in ['frozen','decota']}
                preds.update({k:dict(boxes=v[0],indices=x['predictions']['decota']['indices']) for k,v in variants.items()})
                preds['query_tta']=x['predictions']['decota_s']
                save(f,dict(input=row['input'],parent=row['parent'],frame_ids=x['frame_ids'],predictions=preds,
                    expert=list(x['expert'].values()),called=[k for k in x['keyframe_sets']['main'] if k in x['expert']],pseudo=pseudo,
                    variant_audits={k:v[1] for k,v in variants.items()},GT_used=False,lock_sha256=sha(OUT/'lock.json')))
            print('HISTORY_DONE',g,len(rows),flush=True)


def run(b,g,limit=None):
    import torch
    from threadpoolctl import threadpool_limits
    from methods.decota_refine_v1.predictor import DeCoTARefinePredictor
    from methods.decota_refine_v1.api import spatial_variants
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.dense_expansion_data_v1 import decode_raw
    configure();p=plan();assert (OUT/'SMOKE_AUDIT.json').is_file()
    assert sha(Path(p['expert_snapshot'])/'model.safetensors')==p['expert_sha256']
    m=student(p,b,g);expert=SpatialExpert(p['expert_snapshot'])
    predictor=DeCoTARefinePredictor(m,expert,None,backbone=b,temporal_config=p['temporal_configs'][b][g])
    with threadpool_limits(limits=1):
        for row in p['new'][g][:limit]:
            j=row['ordinal'];f=OUT/'new'/b/g/f'{j:03d}.pt'
            if f.exists():continue
            q=row['input'];frames,ids=decode_raw(q);r=predictor.predict(frames,ids,q,subject=row['student_subject']['subject'],parsed_visual_query=row['visual_query'])
            native=r['runtime'];variants=spatial_variants(native['boxes'],r['pseudo'],ids)
            assert torch.equal(variants['residual'][0],r['boxes'])
            predictions=dict(frozen=dict(boxes=native['boxes'],indices=native['native_indices']),decota=dict(boxes=native['boxes'],indices=r['indices']))
            predictions.update({k:dict(boxes=v[0],indices=r['indices']) for k,v in variants.items()})
            save(f,dict(input=q,parent=row['parent'],frame_ids=ids,predictions=predictions,expert=r['expert'],
                called=[z['position'] for z in r['expert']],pseudo=r['pseudo'],variant_audits={k:v[1] for k,v in variants.items()},
                runtime=native,timing=r['timing'],spatial_backward_calls=0,spatial_optimizer_steps=0,state_reset_checked=True,
                GT_used=False,lock_sha256=sha(OUT/'lock.json')))
            status(OUT/'progress.json',dict(backbone=b,group=g,done=j+1,total=64,unix=time.time()))
            print('NEW',b,g,j+1,64,'pseudo',len(r['pseudo']),'seconds',round(r['timing']['wall_seconds'],2),flush=True)
            del frames,r,native,variants,predictions;gc.collect();torch.cuda.empty_cache()


def seal(part):
    p=plan();receipts=[]
    for b in (['tastvg'] if part=='history' else ['tastvg','tubedetr']):
        for g,rows in p[part].items():
            for j,row in enumerate(rows):
                f=OUT/part/b/g/f'{j:03d}.pt';x=load(f)
                assert not x['GT_used'] and x['lock_sha256']==sha(OUT/'lock.json')
                receipts.append(dict(backbone=b,group=g,ordinal=j,path=str(f),sha256=sha(f),source=row['input']['source']))
    write(OUT/f'{part}_barrier.json',dict(receipts=receipts,GT_used=False,unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','history','run','seal']);ap.add_argument('--backbone');ap.add_argument('--group');ap.add_argument('--part');ap.add_argument('--limit',type=int);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='smoke':smoke()
    elif a.stage=='history':history()
    elif a.stage=='run':run(a.backbone,a.group,a.limit)
    else:seal(a.part)
