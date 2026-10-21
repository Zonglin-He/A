"""Four already-captured development forwards: verify exact visual/RGB cache reuse."""
import json,sys,time,gc
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.boxfixed_tuning.capture import OUT
from scripts.boxfixed_tuning.cache import RGBCache,ReusingTeacher
from scripts.tube_fusion.core import head_logits
from scripts.tube_temporal.run import read,atomic

@torch.no_grad()
def main():
    from stvg.runtime import setup,lease,load_source
    from stvg.predictor import Predictor
    from stvg.visual_cache import VisualCache
    setup();gpu=lease();proof=[];lock=read(OUT/'capture_lock.json')
    selected=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection']
    dev=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()]
    try:
        for ds in ('hc2','vidstg'):
            if ds=='hc2':
                r=selected[ds][0]['record'];key=r['id'];tr=ROOT/'outputs/interval_pool/recheck_hc2'
                reference=torch.load(tr/'capture'/f'{key}.pt',map_location='cpu',weights_only=False)
                logits=torch.load(tr/'posteriors'/f'{key}.pt',map_location='cpu',weights_only=False)['logits']
            else:
                r=next(v for v in dev if v['dataset']==ds);key=r['id']
                reference=torch.load(OUT/'captures'/ds/f'{key}.pt',map_location='cpu',weights_only=False);logits=reference['logits']
            model=Predictor(ds);model.source=load_source(model.config)
            expert=ReusingTeacher(OUT/'regression_expert_cache',lock['runtime_fingerprint'],
                previous=[ROOT/'outputs/interval_pool/teacher_cache',OUT/'teacher_cache']);model.teacher_service=expert
            visual=VisualCache(4*2**30);rgb=RGBCache()
            try:
                for repeat in range(2):
                    frames,ids,rc=rgb.get(r['metadata'],ds);torch.manual_seed(20260920);np.random.seed(20260920)
                    before=visual.stats();start=time.perf_counter()
                    with visual.context(model.source,frames,ids):cap=model.capture(frames,ids,r['metadata'],temporal_mode='native224',export_replay=True)
                    tc=cap['temporal_cache'];head=model.source.temp_embed.cuda()
                    current,_=head_logits(head,[z['hidden']for z in tc['offsets']],[z['native_logits']for z in tc['offsets']]);head.cpu()
                    assert np.array_equal(cap['Source'],reference['Source'])and np.array_equal(cap['E'],reference['E'])
                    pos=reference.get('positions8',reference.get('planned_positions'));assert list(cap['planned_positions'])==list(pos)
                    assert all(torch.equal(a,b)for a,b in zip(current,logits))
                    geometry=cap['geometry'].state_dict();refgeometry=reference['geometry'];checked=0
                    for name,v in refgeometry.items():
                        if torch.is_tensor(v):assert torch.equal(geometry[name],v),name;checked+=1
                    assert cap['cost']['ReferDINO_query_forwards']==0 and cap['cost']['Source448_calls']==0
                    after=visual.stats();proof.append(dict(dataset=ds,repeat=repeat,RGB_cache_hit=rc['hit'],
                        visual_hits=after['hits']-before['hits'],visual_misses=after['misses']-before['misses'],
                        Source_E_positions_six_layers_all_geometry_bitexact=True,geometry_tensors_checked=checked,
                        expert_forwards=0,seconds=time.perf_counter()-start,GT_read=False))
                    del cap,tc,current,geometry,frames;gc.collect();torch.cuda.empty_cache()
            finally:expert.close()
            del model,visual,rgb;gc.collect();torch.cuda.empty_cache()
        assert [p['visual_hits']>0 for p in proof]==[False,True,False,True]
        atomic(ROOT/'results/boxfixed_tuning_v1/cache_regression.json',dict(status='pass',rows=proof,
            additional_existing_query_native_captures=4,new_experts=0,GT_read=False,
            scope='Engineering-only old-query miss/hit numeric regression, not a new accuracy experiment or speed comparison.'))
        print(json.dumps(proof,indent=2))
    finally:gpu.close()

if __name__=='__main__':main()
