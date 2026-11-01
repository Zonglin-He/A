"""Single-factor source-score calibration comparison; fixed frozen evidence bank."""
import numpy as np
def prior_score(z,cfg):
    return np.log(np.maximum(z['scores'],1e-12)) if cfg['prior']=='log' else z['scores'].astype(float)
def soft_score(z,e,cfg,vision=True,relation=True,token=True,restricted=False):
    v=e['vision'].copy();rel=e['relation'].copy();pos=e['position'].copy()
    if restricted:
        allowed={int(z['scores'].argmax())}|{p['qB_PRIVATE'] for p in e['legacy']['pairs']};mask=np.ones(256,bool);mask[list(allowed)]=False;v[mask]=0;rel[mask]=0;pos[mask]=0
    return prior_score(z,cfg)+cfg['gamma']*e['token']*token+cfg['vision']*v*vision+cfg['relation']*(pos+rel*relation)
