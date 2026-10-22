"""Authorized HC64 stable-crop temporal trial; all predictions precede labels."""
import os,sys,json,time,gc,hashlib,argparse,fcntl,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.local224.core import crop_windows,crop_stats,single_posterior
from scripts.local224.model import forward
from scripts.branch_decomposition.supplement import read,write,sha,identity
OUT=ROOT/'outputs/local224_v1';PUBLIC=ROOT/'results/local224_v1'
NAMES=['Source','Global6','GlobalPlusFull224','GlobalPlusLocal224','GlobalPlusShifted224','Strong']

def inactive(x):
    try:
        p=Path('/proc')/str(x['pid']);a=(p/'stat').read_text().rsplit(')',1)[1].split()
        return int(a[19])!=x['startticks'] or a[0]=='Z'
    except (FileNotFoundError,KeyError):return True

def boundary():
    s=read(ROOT/'outputs/boxfixed_full_v1/state.json');assert s['status']=='paused_by_user'
    assert all(inactive(x)for x in [s['controller'],*s.get('children',[])])
    saved=read(ROOT/s['pause_receipt']);assert saved['all_owned_exited'] and saved['completed']==s['paused_completed']

def validate():
    p=read(OUT/'plan.json');v={k:x for k,x in p.items()if k!='fingerprint'}
    assert hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()==p['fingerprint']
    for rel,h in p['sources'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in p['assets'].items():assert sha(ROOT/rel)==h,rel
    assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['full_plan_sha256']
    for item in p['items']:
        for k,path in item['inputs'].items():assert sha(ROOT/path)==item['hashes'][k]
    return p

def prepare():
    assert not(OUT/'plan.json').exists();boundary()
    before=read(OUT/'state.json');write(OUT/'deferred_registration.json',before)
    full=read(ROOT/'outputs/boxfixed_full_v1/plan.json');branch=read(ROOT/'outputs/branch_decomposition_v1/plan.json')
    records={r['id']:r for r in [json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()]}
    items=[]
    from scripts.tube_fusion.core import restore_geometry
    from stvg.spatial import evidence_dp
    for previous in branch['items']:
        record=records[previous['id']];assert record['dataset']=='hc2'
        cap=torch.load(ROOT/previous['inputs']['capture'],map_location='cpu',weights_only=False)
        g=restore_geometry(cap['geometry']);track=evidence_dp(g.for_full_timeline())['final'].numpy()
        w,h=record['metadata']['width'],record['metadata']['height'];real,shift=crop_windows(track,w,h)
        inputs={k:previous['inputs'][k]for k in ['capture','G','prediction']}
        items.append(dict(id=previous['id'],query=previous['anonymous'],record=record,inputs=inputs,hashes={k:sha(ROOT/v)for k,v in inputs.items()},real=list(real),shifted=list(shift),crop_stats=crop_stats(track,real,shift,w,h)))
    assert len(items)==64
    sources={**full['sources'],**{str(f.relative_to(ROOT)):sha(f)for f in Path(__file__).parent.glob('*.py')}}
    # Pin the exact safe generic helpers as well as their frozen transitive imports.
    for rel in ['scripts/branch_decomposition/supplement.py','scripts/branch_decomposition/lightweight_design.py']:
        sources[rel]=sha(ROOT/rel)
    p=dict(schema='stable_local224_hc64_v1',authorization='User explicitly paused and saved full evaluation to prioritize this temporal trial now; earlier deferred ordering superseded.',items=items,sources=sources,assets=full['assets'],full_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),labels=branch['labels'],config=read(ROOT/'configs/default.json')['datasets']['hc2'],GT_read=False,
        protocol=dict(context=1.5,offset=0,mix=.5,window='Full timeline BoxFixed union, one integer rectangle, no event conditioning. Same-size farthest legal corner control.',preprocess='Original fixed height224, original capped aspect width/interpolation/normalization; not arbitrary short-side resizing.',budgets=dict(local_native_views=128,engineering_full224_view=1,new_teachers=0,updates=0),positive_control='One original whole224 offset0 temporal logits within preregistered1e-4 FP32 and exactly same single-offset MBR; preserve discrepancy and stop on failure.',readout='Original physical posterior and MBR, one final original BoxFixed; Strong is cached reference only. No extra448, no new teacher, no sweep. Original native internal selection retained, no internal head used as a new score.',evaluation='Seal all64 matched predictions before CPU labels. Main local vs full224 and shifted; paired t/fullv/fixedspacev, boundary, all signed/severe losses and actual new stage costs.',exposure='HC64 repeatedly exposed development; no adoption or independent confirmation claim.'))
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(OUT/'plan.json',p)
    for rel in sources:
        q=OUT/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
    write(OUT/'state.json',dict(status='prepared',fingerprint=p['fingerprint'],completed=0,total=64));print('prepared',p['fingerprint'])

@torch.no_grad()
def execute(resume=False):
    p=validate();boundary();assert not(OUT/'sealed.json').exists(),'Do not rerun sealed predictions'
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(OUT/'state.json')
    if state['status']!='prepared':assert resume and inactive(state.get('process',{}))
    from stvg.runtime import setup,lease,load_source
    from stvg.decode import decode
    from stvg.inputs import validate_input
    from stvg.query import QuerySubjectParser
    from stvg.temporal import final_mbr
    from scripts.tube_fusion.core import restore_geometry
    from stvg.spatial import evidence_dp
    setup();gpu=lease();state=dict(status='running',fingerprint=p['fingerprint'],process=identity(),completed=0,total=64);write(OUT/'state.json',state)
    model=None;files=[];begin=time.perf_counter()
    try:
        tick=time.perf_counter();model=load_source(p['config']);torch.cuda.synchronize();model_load=time.perf_counter()-tick
        tick=time.perf_counter();parser=QuerySubjectParser(ROOT/'checkpoints/stanza');parser_load=time.perf_counter()-tick
        # Before changed-input predictions, verify the one-offset temporal adapter.
        if not(OUT/'engineering.json').exists():
            item=p['items'][0];record=item['record'];frames,ids=decode(record['metadata'],'hc2');validate_input(frames,ids,record['metadata'])
            cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False)
            row=dict(input=record['metadata'],frame_ids=ids,parses=dict(subject=parser(record['metadata']['caption'])['subject']))
            torch.manual_seed(20260920);np.random.seed(20260920);got=forward(model,frames,row)
            saved=cap['temporal_cache']['offsets'][0]['native_logits'];error=float((got['logits']-saved).abs().max())
            q=single_posterior(got['ids'],got['logits'].numpy(),ids);before=single_posterior(got['ids'],saved.numpy(),ids)
            evidence=dict(max_abs_logit_error=error,bitwise_logits=torch.equal(got['logits'],saved),same_MBR=final_mbr(ids,q)['interval']==final_mbr(ids,before)['interval'],tolerance=1e-4,costs=got['costs'],module_calls=got['module_calls'],native_time_passes=got['native_time_passes'],extra_whole224_engineering_calls=1,GT_read=False)
            torch.save(dict(got=got,saved=saved,GT_read=False),OUT/'engineering_raw.pt');write(OUT/'engineering.json',evidence)
            assert error<=1e-4 and evidence['same_MBR'],'Single-offset temporal adapter failed positive control; preserve evidence'
            del frames,cap,got;gc.collect();torch.cuda.empty_cache()
        control=read(OUT/'engineering.json');assert control['max_abs_logit_error']<=1e-4 and control['same_MBR']
        for item in p['items']:
            key=item['query'];rp=OUT/'receipts'/f'{key}.json';pp=OUT/'predictions'/f'{key}.npz';cp=OUT/'posteriors'/f'{key}.pt'
            if rp.exists():
                rec=read(rp);assert rec['fingerprint']==p['fingerprint'] and sha(pp)==rec['prediction_sha256'] and sha(cp)==rec['posterior_sha256'];files.append(rec);continue
            assert not pp.exists() and not cp.exists(),'Preserve unreceipted evidence instead of overwriting'
            query_tick=time.perf_counter();record=item['record'];tick=time.perf_counter();frames,ids=decode(record['metadata'],'hc2');validate_input(frames,ids,record['metadata']);decode_time=time.perf_counter()-tick
            tick=time.perf_counter();subject=parser(record['metadata']['caption'])['subject'];parse_time=time.perf_counter()-tick
            row=dict(input=record['metadata'],frame_ids=ids,parses=dict(subject=subject));cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False);low=torch.load(ROOT/item['inputs']['G'],map_location='cpu',weights_only=False)
            with np.load(ROOT/item['inputs']['prediction'])as old:
                values=dict(Source=old['Source_Native224'],Source_interval=old['Source_Native224_interval'],Strong=old['Strong_Reference'],Strong_interval=old['Strong_Reference_interval'],frame_ids=old['frame_ids'],positions8=old['positions8'],FixedSpatial=old['Strong_Reference'])
                expected_global=old['Global6'].copy();expected_interval=old['Global6_interval'].tolist()
            qG=low['final']['Global6'];qF=single_posterior(cap['records'][0]['frame_ids'],cap['temporal_cache']['offsets'][0]['native_logits'].numpy(),ids)
            qs=dict(Global6=qG,GlobalPlusFull224=.5*qG+.5*qF);local={};time_cost={};torch.cuda.reset_peak_memory_stats()
            for label,window in [('Local224',item['real']),('Shifted224',item['shifted'])]:
                tick=time.perf_counter();x0,y0,x1,y1=window;crop=np.ascontiguousarray(frames[:,y0:y1,x0:x1]);crop_time=time.perf_counter()-tick
                torch.manual_seed(20260920);np.random.seed(20260920);tick=time.perf_counter();out=forward(model,crop,row);torch.cuda.synchronize();network_wall=time.perf_counter()-tick
                tick=time.perf_counter();q=single_posterior(out['ids'],out['logits'].numpy(),ids);post_time=time.perf_counter()-tick
                qs['GlobalPlus'+label]=.5*qG+.5*q
                time_cost[label]=dict(crop_seconds=crop_time,network_wall_seconds=network_wall,posterior_seconds=post_time,input_shape=out['input_shape'],module_seconds=out['costs'],module_calls=out['module_calls'],native_time_passes=2)
                local[label]=dict(logits=out['logits'],ids=out['ids']);del out,crop;gc.collect();torch.cuda.empty_cache()
            geom=restore_geometry(cap['geometry']);tick=time.perf_counter()
            for name,q in qs.items():
                interval=final_mbr(ids,q)['interval'];values[name]=evidence_dp(geom.for_interval(interval))['final'].numpy();values[name+'_interval']=interval
            readout_seconds=time.perf_counter()-tick
            assert np.array_equal(values['Global6'],expected_global) and values['Global6_interval']==expected_interval
            pp.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**values);torch.save(dict(local=local,qs=qs,GT_read=False),cp)
            rec=dict(query=key,fingerprint=p['fingerprint'],prediction_path=str(pp.relative_to(OUT)),prediction_sha256=sha(pp),posterior_path=str(cp.relative_to(OUT)),posterior_sha256=sha(cp),decode_seconds=decode_time,parse_seconds=parse_time,readout_all_arms_seconds=readout_seconds,branches=time_cost,crop_stats=item['crop_stats'],peak_GPU_bytes=torch.cuda.max_memory_allocated(),query_wall_seconds=time.perf_counter()-query_tick,GT_read=False,new_expert_calls=0,parameter_updates=0,local224_views=2,Source448_calls=0)
            write(rp,rec);files.append(rec);state['completed']=len(files);state['recent_query_seconds']=float(np.mean([r['query_wall_seconds']for r in files[-10:]]));write(OUT/'state.json',state)
            print(json.dumps(dict(completed=len(files),total=64,recent_seconds=state['recent_query_seconds'])),flush=True)
            del cap,low,geom,frames,values,qs,local;gc.collect();torch.cuda.empty_cache()
        validate();assert len(files)==64
        write(OUT/'sealed.json',dict(fingerprint=p['fingerprint'],files=files,queries=64,GT_read=False,seconds=time.perf_counter()-begin,model_load_seconds=model_load,parser_load_seconds=parser_load,engineering_sha256=sha(OUT/'engineering.json')))
        state['status']='sealed_awaiting_CPU_score';write(OUT/'state.json',state)
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc();write(OUT/'state.json',state);raise
    finally:
        if model is not None:model.cpu()
        gpu.close();lock.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--resume',action='store_true');v=a.parse_args()
    prepare()if v.action=='prepare'else execute(v.resume)
