"""Fixed HC64 native/tube temporal comparison with an isolated legacy reference.

Preparation verifies saved inputs and the old BoxFixed run under its immutable
source snapshot. Inference has no labels interface and imports no scorer.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,subprocess,sys,time,traceback,gc
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/tube_temporal_hc64'
LEGACY=ROOT/'outputs/maintenance/boxfixed_switch_20261021/legacy'
SNAPSHOT=LEGACY.parent/'snapshot.json'
REFERENCE=ROOT/'outputs/box_propagation_hc64'
METHODS=('A_Strong','B_Native224','C_TubeReplay','D_Permuted')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
def savept(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)
def within(base,relative):
    path=(Path(base)/relative).resolve();assert path.is_relative_to(Path(base).resolve());return path

def function_ast_hashes(path,names):
    """Hash only explicitly used metric helpers; source-file hash is locked too."""
    tree=ast.parse(Path(path).read_text());functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert set(names)<=set(functions)
    return {name:hashlib.sha256(ast.dump(functions[name],include_attributes=False).encode()).hexdigest()for name in names}

def metric_compatibility():
    sources={k:v for k,v in read(SNAPSHOT)['sources'].items()if k.startswith('stvg/metrics')}
    for path,digest in sources.items():assert sha(ROOT/path)==digest,'Metric implementation changed: '+path
    names=('fps_value','add_reporting_metrics','finite_mean','main_table')
    current=function_ast_hashes(ROOT/'stvg/evaluation.py',names);legacy=function_ast_hashes(LEGACY/'stvg/evaluation.py',names)
    assert current==legacy,'Used evaluation helper changed'
    return sources,dict(path='stvg/evaluation.py',function_ast_sha256=current,legacy_function_ast_sha256=legacy,
        current_file_sha256=sha(ROOT/'stvg/evaluation.py'),legacy_file_sha256=sha(LEGACY/'stvg/evaluation.py'),AST_excludes_line_attributes=True)

def legacy_validation():
    snapshot=read(SNAPSHOT);assert Path(snapshot['root']).resolve()==LEGACY.resolve()
    for path,digest in snapshot['sources'].items():assert sha(LEGACY/path)==digest,path
    assert sha(LEGACY/'configs/default.json')==snapshot['old_default_sha256']
    assert sha(REFERENCE/'lock.json')==snapshot['locks']['outputs/box_propagation_hc64/lock.json']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('STVG_PROJECT_ROOT',None);env.pop('PYTHONPATH',None)
    code="import json;from scripts.box_propagation.run import validate;v=validate();print(json.dumps({'validated_queries':len(v),'GT_read':False}))"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=LEGACY,env=env,text=True,capture_output=True,check=True)
    answer=json.loads(result.stdout.strip().splitlines()[-1]);assert answer==dict(validated_queries=64,GT_read=False)
    return dict(snapshot_sha256=sha(SNAPSHOT),legacy_sources_verified=len(snapshot['sources']),
        old_validator_executed_under_snapshot=True,validated_queries=64,GT_read=False,
        reference_lock_sha256=sha(REFERENCE/'lock.json'),reference_seal_sha256=sha(REFERENCE/'sealed.json'))

def prepare():
    cfg=read(HERE/'config.json');assert cfg['queries']==64 and cfg['methods']==list(METHODS)and cfg['tube_alpha']==.5 and cfg['tube_context_scale']==1.5
    proof=legacy_validation();old=read(REFERENCE/'lock.json');seal=read(REFERENCE/'sealed.json')
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==old['manifest_sha256']
    records=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()];records=[r for r in records if r['dataset']=='hc2'][:64]
    assert [r['id']for r in records]==old['ids']
    inputs=[]
    for item,record in zip(old['inputs'],records):
        key=item['id'];r=read(REFERENCE/'receipts'/f'{key}.json');entry=seal['files'][key]
        cache=ROOT/'outputs/layout_reward_dev/student_cache'/f'{key}.pt'
        mask=ROOT/'outputs/layout_reward_dev/research/native_masks'/item['mask_path']
        assert sha(cache)==item['cache_sha256']and sha(mask)==item['native_mask_sha256']
        assert sha(REFERENCE/entry['path'])==entry['sha256']and Path(record['metadata']['video_path']).is_file()
        inputs.append(dict(id=key,key=record['key'],cache_path=str(cache.relative_to(ROOT)),cache_sha256=sha(cache),
            mask_path=str(mask.relative_to(ROOT)),mask_sha256=sha(mask),reference_path=entry['path'],reference_sha256=entry['sha256'],
            reference_receipt_path=f'receipts/{key}.json',reference_receipt_sha256=sha(REFERENCE/'receipts'/f'{key}.json'),
            bank_path=r['bank_path'],bank_sha256=r['bank_sha256']))
    from stvg.worker import runtime_fingerprint
    runtime,runtimeproof=runtime_fingerprint('tastvg')
    sources={**runtimeproof['code'],**{str(p.relative_to(ROOT)):sha(p)for p in HERE.glob('*')if p.is_file()and p.suffix in ('.py','.json')and not p.name.startswith('test_')}}
    sources['scripts/state_opd/evaluate.py']=sha(ROOT/'scripts/state_opd/evaluate.py')
    for path,digest in sources.items():assert sha(ROOT/path)==digest
    metric_sources,metric_helpers=metric_compatibility()
    identity=dict(config=cfg,sources=sources,metric_sources=metric_sources,metric_helpers=metric_helpers,runtime_fingerprint=runtime,
        legacy=proof,manifest_sha256=sha(manifest),formal_state_sha256=sha(ROOT/'outputs/formal/state.json'),
        registered_config_sha256=sha(ROOT/'configs/default.json'),ids=[r['id']for r in records],inputs=inputs,GT_read=False)
    lock={**identity,'fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
    if(TRIAL/'lock.json').exists():assert read(TRIAL/'lock.json')==lock,'Do not mix method/code/input identities'
    else:
        atomic(TRIAL/'lock.json',lock);atomic(TRIAL/'legacy_validation.json',proof);atomic(TRIAL/'runtime_fingerprint.json',runtimeproof)
        for relative in sources:
            dest=TRIAL/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
    return cfg,lock,records

def check_sources(lock):
    for path,digest in lock['sources'].items():assert sha(ROOT/path)==digest,path
    snapshot=read(SNAPSHOT);assert sha(SNAPSHOT)==lock['legacy']['snapshot_sha256']
    for path,digest in snapshot['sources'].items():assert sha(LEGACY/path)==digest
    assert sha(REFERENCE/'lock.json')==lock['legacy']['reference_lock_sha256']and sha(REFERENCE/'sealed.json')==lock['legacy']['reference_seal_sha256']
    assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']
    assert sha(ROOT/'configs/default.json')==lock['registered_config_sha256']and sha(ROOT/'data/evaluation/development.jsonl')==lock['manifest_sha256']
    metric_sources,metric_helpers=metric_compatibility();assert metric_sources==lock['metric_sources']and metric_helpers==lock['metric_helpers']

def make_teacher(old,pack,item):
    assert old['GT_read']is False and pack['GT_read']is False and pack['source_cache_sha256']==item['cache_sha256']
    idx=int(pack['selected_query']);teacher=dict(old['teacher']);raw=pack['raw_out']
    assert idx==int(teacher['selected_query'])and torch.equal(raw['pred_boxes'][0,:,idx],teacher['boxes'])
    assert list(pack['positions8'])==list(old['planned_positions'])and list(pack['frame_ids'])==list(old['frame_ids'])
    original_cost=dict(teacher.get('cost',{}));teacher.update(native_mask_logits=raw['pred_masks'][0,:,idx].float().cpu(),
        input_shape=list(pack['input_shape']),nested_mask_all_false=bool(pack['nested_mask_all_false']),
        evidence_binding_verified=True,positions8=list(old['planned_positions']),frame_ids=list(old['frame_ids']),
        query_forwards=0,cache_hit=True,cost={key:0. for key in original_cost},model_load_seconds=0.,
        process_wall_seconds=0.,peak_GPU_bytes=0,evidence_original_cost=original_cost)
    return teacher

def validate_one(item,lock):
    key=item['id'];r=read(TRIAL/'receipts'/f'{key}.json');assert r['GT_read']is False and r['fingerprint']==lock['fingerprint']
    for stem in ('prediction','trace','capture'):
        assert sha(within(TRIAL,r[stem+'_path']))==r[stem+'_sha256']
    assert sha(ROOT/item['cache_path'])==item['cache_sha256']and sha(ROOT/item['mask_path'])==item['mask_sha256']
    assert r['cache_sha256']==item['cache_sha256']and r['mask_sha256']==item['mask_sha256']
    assert r['Source_exact']and r['E_exact']and r['positions_exact']and r['A_reference_exact']
    assert r['Source448_calls']==r['expert_query_forwards']==r['actual_updates']==0
    assert r['extra_decoder_calls']=={'B_Native224':0,'C_TubeReplay':2,'D_Permuted':2}
    assert sha(REFERENCE/item['reference_path'])==item['reference_sha256']==r['reference_sha256']
    assert sha(REFERENCE/item['reference_receipt_path'])==item['reference_receipt_sha256']
    cache=torch.load(ROOT/item['cache_path'],map_location='cpu',weights_only=False)
    capture=torch.load(TRIAL/r['capture_path'],map_location='cpu',weights_only=False)
    trace=torch.load(TRIAL/r['trace_path'],map_location='cpu',weights_only=False)
    assert capture['GT_read']is False and set(trace)==set(METHODS[1:])
    with np.load(TRIAL/r['prediction_path'],allow_pickle=False)as p,np.load(REFERENCE/item['reference_path'],allow_pickle=False)as old:
        expected=set(METHODS)|{m+'_interval'for m in METHODS}|{'Source','E_Ref','frame_ids','positions8'}
        assert set(p.files)==expected
        assert np.array_equal(p['A_Strong'],old['BoxFixed'])and np.array_equal(p['A_Strong_interval'],old['Strong_interval'])
        assert np.array_equal(p['frame_ids'],old['frame_ids'])and np.array_equal(p['positions8'],old['positions8'])
        assert np.array_equal(p['Source'],np.asarray(cache['Source']))and np.array_equal(p['E_Ref'],np.asarray(cache['E']))
        assert np.array_equal(p['Source'],np.asarray(capture['Source']))and np.array_equal(p['E_Ref'],np.asarray(capture['E']))
        assert p['frame_ids'].tolist()==list(cache['frame_ids'])==list(capture['frame_ids'])
        assert p['positions8'].tolist()==list(cache['planned_positions'])==list(capture['planned_positions'])
        for method in METHODS:
            assert p[method].shape==p['A_Strong'].shape and np.isfinite(p[method]).all()and(p[method][:,2:]>0).all()
            interval=p[method+'_interval'];assert interval.shape==(2,)and interval[0]<interval[1]
            assert interval[0]in p['frame_ids']and interval[1]-1 in p['frame_ids']
            if method!='A_Strong':
                out=trace[method];assert out['GT_read']is False and np.array_equal(p[method],np.asarray(out['boxes']))and np.array_equal(interval,np.asarray(out['interval']))
                assert out['cost']==r['costs'][method]and out['cost']['actual_updates']==0
    assert capture['temporal_cache']['GT_read']is False and capture['temporal_cache']['resolution']==224
    assert len(capture['temporal_cache']['offsets'])==2
    C,D=trace['C_TubeReplay'],trace['D_Permuted']
    assert torch.equal(C['auxiliary']['final'],D['auxiliary']['final'])and torch.equal(C['auxiliary']['actions'],D['auxiliary']['actions'])
    for first,second in zip(C['temporal_audit']['priors'],D['temporal_audit']['priors']):
        assert first['frame_ids']==second['frame_ids']and torch.equal(first['reliability'],second['reliability'])and torch.equal(first['legal_visual'],second['legal_visual'])
        for a,b,legal in zip(first['M'],second['M'],first['legal_visual']):assert torch.equal(a[legal].sort().values,b[legal].sort().values)
    for out in (C,D):
        audit=out['temporal_audit'];assert audit['metadata']['alpha']==.5 and audit['metadata']['GT_read']is False
        assert audit['cost']['offset_time_decoder_calls']==[1,1]
        assert audit['cost']['extra_encoder_calls']==audit['cost']['extra_448_calls']==audit['cost']['extra_expert_calls']==audit['cost']['optimizer_steps']==0
        for layer in audit['attention_audit']:
            assert layer['text_max_error']==layer['padding_max_error']==0 and layer['visual_mass_max_error']<2e-6
    return r

def validate(trial=TRIAL,lock=None):
    assert Path(trial).resolve()==TRIAL.resolve();lock=read(TRIAL/'lock.json')if lock is None else lock;check_sources(lock)
    seal=read(TRIAL/'sealed.json');assert seal['queries']==64 and seal['GT_read']is False and seal['fingerprint']==lock['fingerprint']and seal['lock_sha256']==sha(TRIAL/'lock.json')
    assert set(seal['files'])==set(lock['ids']);receipts={}
    for item in lock['inputs']:
        key=item['id'];r=validate_one(item,lock);entry=seal['files'][key]
        assert sha(TRIAL/entry['path'])==entry['sha256']and sha(TRIAL/entry['receipt_path'])==entry['receipt_sha256'];receipts[key]=r
    return receipts

def execute(cfg,lock,records):
    from stvg.runtime import setup,lease
    from stvg.worker import resource_guard
    from stvg.predictor import Predictor
    from stvg.decode import decode
    from stvg.inputs import validate_input
    from stvg.tensors import detached
    assert not(TRIAL/'run.json').exists()and not(TRIAL/'sealed.json').exists(),'No implicit restart or overwrite'
    setup();gpu=lease();torch.set_num_threads(4);predictor=Predictor('hc2',check=resource_guard);files={};begin=time.perf_counter()
    atomic(TRIAL/'run.json',dict(status='running',PID=os.getpid(),fingerprint=lock['fingerprint'],GT_read=False))
    try:
        for i,(record,item)in enumerate(zip(records,lock['inputs'])):
            resource_guard();key=item['id'];start=time.perf_counter();tick=start
            old=torch.load(ROOT/item['cache_path'],map_location='cpu',weights_only=False);pack=torch.load(ROOT/item['mask_path'],map_location='cpu',weights_only=False)
            teacher=make_teacher(old,pack,item);cache_IO=time.perf_counter()-tick;tick=time.perf_counter()
            assert teacher['text_sha256']==hashlib.sha256(' '.join(record['metadata']['caption'].lower().split()).encode()).hexdigest()
            frames,ids=decode(record['metadata'],'hc2');validate_input(frames,ids,record['metadata']);decode_seconds=time.perf_counter()-tick
            torch.manual_seed(cfg['source_rng_seed']);np.random.seed(cfg['source_rng_seed']);torch.cuda.reset_peak_memory_stats()
            cache=predictor.capture(frames,ids,record['metadata'],temporal_mode='native224',export_replay=True,teacher_override=teacher)
            assert np.array_equal(np.asarray(cache['Source']),np.asarray(old['Source'])),'Fresh native Source differs from reference'
            assert np.array_equal(np.asarray(cache['E']),np.asarray(old['E'])),'Fresh ERef differs from reference'
            assert list(cache['planned_positions'])==list(old['planned_positions'])
            assert cache['cost']['Source448_calls']==0 and cache['cost']['ReferDINO_query_forwards']==0
            del frames,pack;outputs={};costs={}
            for name,mode,permute in [('B_Native224','native224',False),('C_TubeReplay','tube_replay',False),('D_Permuted','tube_replay',True)]:
                out=predictor.finish(cache,mode,permuted=permute);outputs[name]=detached(out,'cpu');costs[name]=out['cost']
                assert out['cost']['extra_decoder_calls']==(0 if name=='B_Native224'else 2)
            with np.load(REFERENCE/item['reference_path'],allow_pickle=False)as ref:
                values=dict(A_Strong=ref['BoxFixed'].copy(),A_Strong_interval=ref['Strong_interval'].copy(),
                    Source=np.asarray(cache['Source']),E_Ref=np.asarray(cache['E']),frame_ids=np.asarray(ids),positions8=np.asarray(cache['planned_positions']))
            for name,out in outputs.items():values[name]=out['boxes'];values[name+'_interval']=out['interval']
            capture=dict(Source=cache['Source'],E=cache['E'],planned_positions=cache['planned_positions'],frame_ids=cache['frame_ids'],
                evidence=cache['evidence'].state_dict(),geometry=cache['geometry'].state_dict(),temporal_cache=cache['temporal_cache'],
                records=cache['records'],cost=cache['cost'],GT_read=False)
            tick=time.perf_counter();cp=TRIAL/'captures'/f'{key}.pt';tp=TRIAL/'traces'/f'{key}.pt';savept(cp,detached(capture,'cpu'));savept(tp,outputs)
            pp=TRIAL/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True)
            with pp.with_suffix('.tmp').open('wb')as f:np.savez(f,**values)
            os.replace(pp.with_suffix('.tmp'),pp);write_seconds=time.perf_counter()-tick
            r=dict(fingerprint=lock['fingerprint'],prediction_path=str(pp.relative_to(TRIAL)),prediction_sha256=sha(pp),
                trace_path=str(tp.relative_to(TRIAL)),trace_sha256=sha(tp),capture_path=str(cp.relative_to(TRIAL)),capture_sha256=sha(cp),
                cache_sha256=item['cache_sha256'],mask_sha256=item['mask_sha256'],reference_sha256=item['reference_sha256'],
                Source_exact=True,E_exact=True,positions_exact=True,A_reference_exact=True,Source448_calls=0,expert_query_forwards=0,actual_updates=0,
                extra_decoder_calls={m:costs[m]['extra_decoder_calls']for m in costs},costs=costs,capture_cost=cache['cost'],
                cache_hits=dict(expert=True,spatial_reference=True,temporal_replay=True,native224=False,RGB_decode=False),
                timings=dict(input_cache_IO_seconds=cache_IO,RGB_decode_seconds=decode_seconds,output_IO_seconds=write_seconds),
                total_seconds=time.perf_counter()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),GT_read=False)
            rp=TRIAL/'receipts'/f'{key}.json';atomic(rp,r);validate_one(item,lock)
            files[key]=dict(path=str(pp.relative_to(TRIAL)),sha256=sha(pp),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp))
            atomic(TRIAL/'progress.json',dict(completed=i+1,total=64,seconds=time.perf_counter()-begin,GT_read=False));print(json.dumps(dict(completed=i+1,total=64)),flush=True)
            predictor.source.cpu();assert not any(p.requires_grad or p.grad is not None for p in predictor.source.parameters())
            del cache,old,teacher,outputs,values,capture;gc.collect();torch.cuda.empty_cache()
        atomic(TRIAL/'sealed.json',dict(queries=64,fingerprint=lock['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),files=files,GT_read=False));validate()
        atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,fingerprint=lock['fingerprint'],seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(TRIAL/'failure.json',dict(completed=len(files),fingerprint=lock['fingerprint'],traceback=traceback.format_exc(),GT_read=False));raise
    finally:
        if predictor.source is not None:predictor.source.cpu()
        gpu.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    cfg,lock,records=prepare()
    if args.run:execute(cfg,lock,records)
    else:print(json.dumps(dict(status='prepared_not_run',fingerprint=lock['fingerprint'])))
