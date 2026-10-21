"""One existing-query engineering regression and raw-input cache-miss check.

No GT and no new evaluation sample. Separate from the fixed64 prediction run.
"""
import gc,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_temporal.run import make_teacher,sha,atomic,savept
from stvg.runtime import setup,lease
from stvg.decode import decode
from stvg.predictor import Predictor
from stvg.tube_replay import replay_temporal

def main():
    dest=ROOT/'outputs/checks/boxfixed_tube';assert not(dest/'completed.json').exists(),'No repeated timing runs'
    setup();lock=lease();begin=time.perf_counter();L=json.loads((ROOT/'outputs/box_propagation_hc64/lock.json').read_text());item=L['inputs'][0];key=item['id']
    oldp=ROOT/'outputs/layout_reward_dev/student_cache'/f'{key}.pt';mp=ROOT/'outputs/layout_reward_dev/research/native_masks'/item['mask_path']
    assert sha(oldp)==item['cache_sha256']and sha(mp)==item['native_mask_sha256']
    old=torch.load(oldp,map_location='cpu',weights_only=False);pack=torch.load(mp,map_location='cpu',weights_only=False)
    teacher=make_teacher(old,pack,item);records=[json.loads(x)for x in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if x.strip()]
    row=next(r for r in records if r['id']==key);frames,ids=decode(row['metadata'],'hc2')
    torch.manual_seed(20260920);np.random.seed(20260920);p=Predictor('hc2')
    prepared=dest/'prepared_native_capture.pt'
    if prepared.exists():
        from stvg.runtime import load_source
        cap=torch.load(prepared,map_location='cpu',weights_only=False);p.source=load_source(p.config);p.source.cpu()
    else:cap=p.capture(frames,ids,row['metadata'],temporal_mode='native224',export_replay=True,teacher_override=teacher)
    if not np.array_equal(cap['Source'],np.asarray(old['Source'])):
        savept(dest/'failed_source_capture.pt',{'Source':cap['Source'],'oldSource':old['Source'],'spatial_inputs':cap['spatial_inputs'],'temporal_cache':cap['temporal_cache'],'GT_read':False})
        raise AssertionError('Source changed maxabs='+str(np.abs(cap['Source']-np.asarray(old['Source'])).max()))
    assert np.array_equal(cap['E'],np.asarray(old['E'])),'ERef changed'
    from stvg.evidence import SourceEvidence
    from stvg.spatial import build_geometry
    cap['evidence']=SourceEvidence.from_capture(cap['spatial_inputs'],ids,cap['planned_positions'],teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query'])
    cap['geometry']=build_geometry(cap['evidence'],cap['E'],teacher['boxes'])
    savept(dest/'prepared_capture.pt',cap)
    r=cap['geometry'].temporal_reliability();zero=replay_temporal(p.source,cap['temporal_cache'],cap['E'],r,alpha=0.)
    assert zero['interval']==cap['intervals']['Native']and np.array_equal(zero['posterior'],cap['posterior']['Native'])
    # One existing-query448 reference measurement; no new224 encode here.
    from stvg.temporal_capture import frozen_temporal_forward
    from stvg.temporal import probabilities,mass_table,final_mbr
    from stvg.inputs import resolution
    from stvg.query import QuerySubjectParser
    from stvg.tensors import detached
    p.parser=QuerySubjectParser(ROOT/'checkpoints/stanza');parses={'subject':p.parser(row['metadata']['caption'])['subject']}
    p.source.temp_embed.cuda();layers=[p.source.temp_embed(v['hidden'].cuda()).detach().cpu()for v in cap['temporal_cache']['offsets']]
    assert all(torch.equal(layer[-1],v['native_logits'])for layer,v in zip(layers,cap['temporal_cache']['offsets']))
    low=[mass_table(probabilities(cap['records'],[v[k]for v in layers]))[1]for k in range(6)]
    p.source.cuda();tick=time.perf_counter()
    with resolution(448):batch,recs,high=frozen_temporal_forward(p.source,frames,{'input':row['metadata'],'frame_ids':ids,'parses':parses},audit=False)
    U,hq=mass_table(probabilities(recs,detached(high.zero['logits'],'cpu')));q=.5*hq+.5*np.mean(low,axis=0)
    cap['intervals']['Strong']=final_mbr(U,q)['interval'];cap['posterior']['Strong']=q;torch.cuda.synchronize()
    cap['cost']['Source448_seconds']=time.perf_counter()-tick;cap['cost']['Source448_calls']=2
    del batch,high,layers,low;p.source.cpu();gc.collect();torch.cuda.empty_cache()
    savept(dest/'post_strong_capture.pt',cap)
    A=p.finish(cap,'strong');reference=ROOT/'outputs/box_propagation_hc64/predictions'/f'{key}.npz'
    with np.load(reference)as f:
        assert np.array_equal(A['boxes'],f['BoxFixed'])and np.array_equal(A['interval'],f['Strong_interval'])
    B=p.finish(cap,'native224');C=p.finish(cap,'tube_replay');D=p.finish(cap,'tube_replay',permuted=True)
    for c,d in zip(C['temporal_audit']['priors'],D['temporal_audit']['priors']):
        assert torch.equal(torch.sort(c['M'],dim=-1).values,torch.sort(d['M'],dim=-1).values)
        assert torch.equal(c['reliability'],d['reliability'])and torch.equal(c['legal_visual'],d['legal_visual'])
    proof=dict(existing_query=True,GT_read=False,Source_ERef_positions_exact=True,BoxFixed_Strong_exact=True,alpha0_native_logits_hidden_posterior_interval_bitexact=True,
        C_D_same_auxiliary_boxes=torch.equal(C['auxiliary']['final'],D['auxiliary']['final']),C_D_area_reliability_padding_preserved=True,
        native_saved_capture_cost={**cap['cost'],'Source448_seconds':0.,'Source448_calls':0},Strong_reference_additional448={'seconds':cap['cost']['Source448_seconds'],'offset_calls':2,'scope':'separate existing-query448 branch and Strong postprocess; not a Strong end-to-end wall measurement'},B_cost=B['cost'],C_cost=C['cost'],D_cost=D['cost'],alpha0_cost=zero['cost'])
    assert proof['C_D_same_auxiliary_boxes'];savept(dest/'reference_capture.pt',dict(Source=cap['Source'],E=cap['E'],temporal_cache=cap['temporal_cache'],GT_read=False))
    resident_source=p.source;del cap,zero,A,B,C,D,old,pack;p.source.cpu();del p;gc.collect();torch.cuda.empty_cache()
    # Fresh request with no Source/teacher feature or evidence cache injection.
    tick=time.perf_counter();torch.manual_seed(20260920);np.random.seed(20260920);fresh=Predictor('hc2',source=resident_source)
    raw=fresh.predict(frames,ids,row['metadata'],key,temporal_mode='tube_replay');torch.cuda.synchronize()
    assert raw['capture_cost']['Source448_calls']==0 and raw['capture_cost']['ReferDINO_query_forwards']==1
    assert raw['cost']['extra_decoder_calls']==2 and raw['cost']['actual_updates']==0
    assert np.array_equal(raw['Source'],np.asarray(torch.load(oldp,map_location='cpu',weights_only=False)['Source']))
    proof.update(raw_input_independent=True,cache_miss_scope='raw RGB and text, no feature/evidence cache; frozen Source weights already resident on CPU',cache_miss_total_wall_seconds=time.perf_counter()-tick,cache_miss_capture_cost=raw['capture_cost'],cache_miss_finish_cost=raw['cost'],
        fresh_expert_calls=1,raw_448_calls=0,network_frozen=not any(v.requires_grad or v.grad is not None for v in fresh.source.parameters()),seconds=time.perf_counter()-begin)
    savept(dest/'raw_output.pt',{**raw,'GT_read':False});atomic(dest/'completed.json',proof)
    public={k:v for k,v in proof.items()if k not in ('native_saved_capture_cost','cache_miss_capture_cost')}
    public['native_saved_capture_cost']={k:v for k,v in proof['native_saved_capture_cost'].items()if isinstance(v,(int,float,bool))}
    public['cache_miss_capture_cost']={k:v for k,v in proof['cache_miss_capture_cost'].items()if isinstance(v,(int,float,bool))}
    atomic(ROOT/'results/boxfixed_native_smoke.json',public);print(json.dumps(public,indent=2))
    fresh.source.cpu();lock.close()
if __name__=='__main__':main()
