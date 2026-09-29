"""Independent NumPy affinity/area projection/probability/moment geometry readback; no new model run."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import load,read,write,sha
from scripts.run_tastvg_spatial_propagation_s05_v2 import OUT,S0,verify

def run():
    p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');count=0;maxerr=0.;areas=0;boxes=0
    for f,hsh in bar['dense_files'].items():
        assert sha(OUT/f)==hsh;d=load(OUT/f);suffix=Path(f).relative_to('dense');e=load(S0/'expert'/suffix);x=load(S0/'capture'/suffix)
        gh,gw=d['diagnostics']['grid'];n=gh*gw;pos=e['positions'];t=len(x['frame_ids'])
        a=np.stack([x['views'][i%2]['H'][:n,i//2].numpy() for i in range(t)]).astype(float)
        if e['masks'] is not None:
            masks=e['masks'];ih,iw=masks.shape[-2:];occ=np.zeros((len(pos),gh,gw))
            for yy in range(gh):
                for xx in range(gw):occ[:,yy,xx]=masks[:,int(np.floor(yy*ih/gh)):int(np.ceil((yy+1)*ih/gh)),int(np.floor(xx*iw/gw)):int(np.ceil((xx+1)*iw/gw))].mean((1,2))
            np.testing.assert_allclose(occ.reshape(len(pos),n),d['reference_occupancy'].numpy(),atol=1e-7,rtol=0);areas+=1
            fg=occ.reshape(-1)>=.5;ref=a[pos].reshape(-1,a.shape[-1])
            if fg.any() and (~fg).any():
                def unit(z):return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
                z=unit(a);scores=[]
                for bank in [ref[fg],ref[~fg]]:scores.append(.5*(z@unit(bank.mean(0)))+.5*np.max(z@unit(bank).T,axis=-1))
                prob=1/(1+np.exp(scores[1]-scores[0]))
                err=float(np.max(abs(prob-d['probability'].numpy())));assert err<2e-6;maxerr=max(maxerr,err);count+=1
                for j in range(2):np.testing.assert_allclose(scores[j],d['affinity_logits'][...,j].numpy(),atol=2e-6,rtol=0)
                yy,xx=np.mgrid[:gh,:gw];coords=np.stack([(xx.ravel()+.5)/gw,(yy.ravel()+.5)/gh],-1)
                for i in range(t):
                    if i in pos:continue
                    weight=prob[i]/prob[i].sum();center=np.sum(weight[:,None]*coords,axis=0);var=np.sum(weight[:,None]*(coords-center)**2,axis=0);size=np.sqrt(12*var)
                    lo=np.clip(center-size/2,0,1);hi=np.clip(center+size/2,0,1);expect=np.r_[(lo+hi)/2,hi-lo]
                    assert d['valid'][i];np.testing.assert_allclose(d['boxes'][i],expect,atol=5e-7,rtol=0);boxes+=1
        assert np.array_equal(d['valid'][pos],e['valid'][pos]);assert np.array_equal(d['boxes'][pos],e['boxes'][pos])
    result=dict(status='pass',cells=96,independent_area_checks=areas,independent_affinity_cells=count,maximum_probability_error=maxerr,independent_propagated_boxes=boxes,reference_boxes_exact=True)
    write(OUT/'PROPAGATION_AUDIT.json',result);print(result)

if __name__=='__main__':run()
