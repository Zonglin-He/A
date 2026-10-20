"""Same single-expert method with explicitly selected dataset configurations."""
import json
from pathlib import Path
import numpy as np
from .predictor import SingleReferDINOOPDPredictor

def output_writeback(E,raw,alpha):
 E=np.asarray(E,dtype=np.float64);raw=np.asarray(raw,dtype=np.float64);alpha=float(alpha)
 assert np.isfinite(E).all()and np.isfinite(raw).all()and(E[:,2:]>0).all()and(raw[:,2:]>0).all()
 if alpha==0:return E.copy()
 if alpha==1:return raw.copy()
 out=np.concatenate((E[:,:2]+alpha*(raw[:,:2]-E[:,:2]),E[:,2:]*np.exp(alpha*np.log(raw[:,2:]/E[:,2:]))),-1)
 assert np.isfinite(out).all()and(out[:,2:]>0).all();return out

class TunedSingleReferDINOOPDPredictor(SingleReferDINOOPDPredictor):
 def __init__(self,dataset='hc2',config=None,source=None,check=lambda:None):
  if config is None:config=json.loads(Path(__file__).with_name('tuned_config_20261020.json').read_text())['datasets'][dataset]
  super().__init__(dataset,config=config,source=source,check=check)
 def adapt(self,cache,key):
  out=super().adapt(cache,key);raw=out['Live'];final=output_writeback(out['E_Ref'],raw,self.config['output_alpha']);return {**out,'Live_raw':raw,'Live':final,'output_alpha':self.config['output_alpha']}
