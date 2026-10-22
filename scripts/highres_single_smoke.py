"""One preregistered existingH002 actual single448 execution, numerical/cost only.

Separate engineering plan, never pooled as a64th/extra accuracy sample.
"""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.highres_priority.run import SUP,validate as validate_main,assert_pause_boundary,inactive,read,write,sha,identity
OUT=ROOT/'outputs/highres_single_smoke_v1'

def prepare():
    assert not(OUT/'plan.json').exists();p=read(SUP/'plan.json');item=p['items'][0];assert item['anonymous']=='H002'
    sources={**p['sources'],'scripts/highres_single_smoke.py':sha(Path(__file__)),'scripts/local224/model.py':sha(ROOT/'scripts/local224/model.py')}
    q=dict(schema='single448_engineering_v1',item=item,sources=sources,assets=p['assets'],config=p['config'],parent_plan_sha256=sha(SUP/'plan.json'),protocol='Exactlyone existingH002 offset0 at448 through actual single-view adapter. NoGT/newquery/teacher/updates. After main63 seal+GPU exit, before independent labels. Logits and physical posterior must recover parent offset0 bitwise. Save actual wall/load/decode and cold one-case limitations; not a population speed claim.')
    q['fingerprint']=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest();write(OUT/'plan.json',q);print('prepared_single_engineering',q['fingerprint'])

@torch.no_grad()
def execute():
    p=read(OUT/'plan.json');v={k:x for k,x in p.items()if k!='fingerprint'};assert hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()==p['fingerprint']
    assert sha(SUP/'plan.json')==p['parent_plan_sha256'];seal=read(SUP/'sealed.json');assert seal['queries']==63 and seal['GT_read']is False and inactive(read(SUP/'state.json')['process']);assert not(OUT/'receipt.json').exists()
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
    for f,h in p['assets'].items():assert sha(ROOT/f)==h,f
    assert_pause_boundary()
    from stvg.runtime import setup,lease,load_source
    from stvg.decode import decode
    from stvg.inputs import resolution,validate_input
    from stvg.query import QuerySubjectParser
    from scripts.local224.model import forward
    setup();gpu=lease();model=None;tick0=time.perf_counter();write(OUT/'state.json',dict(status='running',process=identity(),fingerprint=p['fingerprint']))
    try:
        tick=time.perf_counter();model=load_source(p['config']);torch.cuda.synchronize();load=time.perf_counter()-tick
        item=p['item'];record=item['record'];tick=time.perf_counter();frames,ids=decode(record['metadata'],'hc2');validate_input(frames,ids,record['metadata']);decode_time=time.perf_counter()-tick
        tick=time.perf_counter();parser=QuerySubjectParser(ROOT/'checkpoints/stanza');subject=parser(record['metadata']['caption'])['subject'];parse=time.perf_counter()-tick
        row=dict(input=record['metadata'],frame_ids=ids,parses=dict(subject=subject));torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        with resolution(448):out=forward(model,frames,row)
        torch.cuda.synchronize();network=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()
        previous=torch.load(SUP/'posteriors/H002.pt',map_location='cpu',weights_only=False)
        assert out['ids']==previous['records'][0]['frame_ids']
        equal=torch.equal(out['logits'],previous['logits'][0]);err=float((out['logits']-previous['logits'][0]).abs().max());torch.save(dict(out=out,previous=previous['logits'][0],GT_read=False),OUT/'raw.pt')
        assert equal,'Actual single448 logits did not recover dual-capture offset0; preserve, no tolerance relaxation'
        rec=dict(status='verified',fingerprint=p['fingerprint'],query='H002',new448_views=1,new224=0,new_expert=0,updates=0,GT_read=False,bitwise_logits=equal,max_error=err,network_wall_seconds=network,decode_seconds=decode_time,parse_with_load_seconds=parse,model_load_seconds=load,whole_engineering_wall_seconds=time.perf_counter()-tick0,peak_GPU_bytes=peak,module_calls=out['module_calls'],two_time_passes=out['native_time_passes'],input_shape=out['input_shape'],costs=out['costs'],scope='Actual one-view code path, one existing query only, fresh model/cold startup. Do not present this as mean63 deployment speed or training-free accuracy confirmation.')
        write(OUT/'receipt.json',rec);write(OUT/'state.json',dict(status='completed',process=identity()));print(json.dumps(rec,indent=2))
    finally:
        if model is not None:model.cpu()
        gpu.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);x=a.parse_args();prepare()if x.action=='prepare'else execute()
