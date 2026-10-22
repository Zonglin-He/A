"""Post-seal positive/negative case reinsertion; never selects an output or step."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));from run import R,D,P,B,read,put,sha,verify
import numpy as np,torch
from core import restore_inputs,target
from stvg.runtime import setup,lease,load_source
from stvg.global6 import posterior
from stvg.temporal import final_mbr
from stvg.spatial import SpatialGeometry,evidence_dp
def main():
    plan=verify();receipt=read(D/'score_receipt.json');assert receipt['seal_sha256']==sha(D/'sealed.json');setup();torch.set_num_threads(1);lock=lease();begin=time.perf_counter();model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.eval().requires_grad_(False);decoder=model.ground_decoder.time_decoder;head=model.temp_embed;checks=[]
    try:
        for key in ['RC025','RC063']:
            cap=torch.load(B/'captures'/f'{key}.pt',map_location='cpu',weights_only=False);tr=torch.load(D/'traces'/f'{key}.pt',map_location='cpu',weights_only=False);kw=[restore_inputs(v,'cuda')for v in cap['temporal_cache']['offsets']]
            for kind in ['Raw','Predictive']:
                delta=tr['paths'][kind]['delta'].cuda();logits=[]
                with torch.no_grad():
                    for k in kw:
                        modified=dict(k);modified['query_tgt']=(k['query_tgt'][:1]+delta[None,None,:]).expand_as(k['query_tgt']);logits.append(head(decoder(**modified)).cpu())
                q=posterior(cap['records'],logits,cap['frame_ids']);error=float(abs(q-tr['q'][kind+'Query']).max());assert error<2e-12
                span=final_mbr(cap['frame_ids'],q)['interval'];geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);boxes=evidence_dp(geom.for_interval(span))['final'].numpy()
                with np.load(D/'predictions'/f'{key}.npz',allow_pickle=False)as z:assert np.array_equal(z[kind+'Query_interval'],span)and np.array_equal(z[kind+'Query'],boxes)
                s=tr['feedback']['raw'if kind=='Raw'else'predictive'];expected=target(tr['q']['NativeG6'],tr['feedback']['pairs'],s);assert np.array_equal(expected,tr['q'][kind+'Direct'])
                checks.append(dict(query=key,kind=kind,posterior_error=error,interval_and_spatial_boxes_bitwise=True,target_recomputed_exact=True))
        assert all(not p.requires_grad and p.grad is None for p in model.parameters());put(P/'reinsertion_audit.json',dict(status='passed',scope='Two post-score opposite-sign cases for engineering reinsertion only; no tuning or selection',checks=checks,new_temporal_decoder_forwards=8,new_Source_encoder=0,new_experts=0,seconds=time.perf_counter()-begin,sealed_predictions_unchanged=True,original_plan_sha256=sha(D/'plan.json')));print(json.dumps(checks))
    finally:lock.close()
if __name__=='__main__':main()
