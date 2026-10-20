"""Dataset-selected initialization damping plus the same OPD/output policy."""
import time
import numpy as np
from .predictor import SingleReferDINOOPDPredictor
from .tuned import TunedSingleReferDINOOPDPredictor,output_writeback

class InitializationTunedSingleReferDINOOPDPredictor(TunedSingleReferDINOOPDPredictor):
 def initialize_cached(self,cache):
  # Reconstruct unscaled consensus each time: repeated calls cannot apply gamma twice.
  raw=SingleReferDINOOPDPredictor.initialize_cached(self,cache);gamma=float(self.config.get('initialization_gamma',1.));E=output_writeback(np.asarray(raw['Source'],dtype=np.float64),raw['E'],gamma);meta=dict(raw['initialization']);meta.update(initialization_gamma=gamma,unscaled_delta=meta['delta'],delta=(gamma*np.array(meta['delta'])).tolist());return {**raw,'E':E,'initialization':meta}
 def capture(self,frames,frame_ids,metadata,parses=None):
  raw=super().capture(frames,frame_ids,metadata,parses);tick=time.perf_counter();out=self.initialize_cached(raw);out['cost']['initialization_strength_seconds']=time.perf_counter()-tick;return out
