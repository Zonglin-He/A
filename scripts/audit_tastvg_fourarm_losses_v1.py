"""NumPy readback of consumed hard/ranking objectives from sealed logits."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_temporal_fourarm_v1 import OUT


def run():
    largest={'Hard':0.,'OPD':0.};count=0
    for rel,h in read(OUT/'PREDICTION_BARRIER.json')['files'].items():
        assert sha(OUT/rel)==h;x=load(OUT/rel);e=load(OUT/'c2'/x['condition']/Path(rel).name)
        for arm in ['Hard','OPD']:
            z=x['arms'][arm];bounds=z['bounds'];scores=e['scores']
            for at,pred in [('initial',x['arms']['Frozen']['prediction']),('final',z['prediction'])]:
                values=[];ells=[]
                for logit,bb in zip(pred['logits'],bounds):
                    a=logit.numpy()[0].astype(float);lp=a-a.max(0);lp-=np.log(np.exp(lp).sum(0));prob=np.exp(lp)
                    ell=np.array([lp[s,0]+lp[t,1] for s,t in bb]);ells.append(ell)
                    target=np.exp(-(np.arange(len(a))[:,None]-np.array(bb[e['selected']]))**2/8)+1e-6;target/=target.sum(0)
                    values.append((prob*np.log((prob+1e-6)/target)).sum(1).mean()*2)
                if arm=='Hard':value=float(np.mean(values))
                else:
                    ell=np.mean(ells,0);terms=[np.logaddexp(0,-(ell[i]-ell[j])) for i in range(len(scores)) for j in range(len(scores)) if scores[i]>scores[j]+1e-12];value=float(np.mean(terms)) if terms else 0.
                actual=z['diagnostics'][at+'_loss'];err=abs(value-actual);largest[arm]=max(largest[arm],err);assert err<3e-6+3e-5*abs(actual),(rel,arm,at,value,actual);count+=1
    result=dict(status='pass',scalar_losses=count,max_absolute_error=largest,hard_sigma=2.,hard_temporal_coefficient=2.,student_yaml_sha256=sha(ROOT/'external/TA-STVG/experiments/vidstg.yaml'))
    write(OUT/'LOSS_READBACK.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
