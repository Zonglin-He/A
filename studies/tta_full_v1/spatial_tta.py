"""Independent Vid execution hook: same Source work yields grids and TTA context."""
from full_common import *
import shutil,traceback

def run():
 import torch
 from engine import Engine
 import stvg.runtime as runtime,stvg.native as native
 validate();assert not(D/'vidstg_worker.json').exists();assert not(PREV/'spatial/vidstg_worker.json').exists();old_load=runtime.load_source;old_forward=native.frozen_native_forward;e=Engine('vidstg',loader=old_load,native_forward=old_forward);atomic(D/'vidstg_worker.json',identity());current={}
 # Original source code bytes/function body remain unchanged. Only new wrapper
 # supplies the same model and collects a second exact interface before discard.
 common=ns.suite();core=ns.load_old('spatial_core',R/'studies/map_paper_completion_v1/spatial_core.py',common);frozen=ns.load_old('spatial',R/'studies/map_paper_completion_v1/spatial.py',common);original_derive=core.derive;original_rows=frozen.rows
 def rows(ds):
  assert ds=='vidstg'
  for row in original_rows(ds):current['row']=row;yield row
 def forward(model,frames,row):
  tick=time.perf_counter();batch,recs,base=old_forward(model,frames,row);e.tapped(current['row'],model,base,recs,row['frame_ids'],time.perf_counter()-tick);return batch,recs,base
 def derive(cap,raw):
  old_threads=torch.get_num_threads()
  try:e.process(current['row'])
  finally:torch.set_num_threads(old_threads)
  assert old_threads==4 and not any(p.requires_grad for p in e.load().parameters());return original_derive(cap,raw)
 frozen.rows=rows;core.derive=derive;runtime.load_source=lambda config:e.load();native.frozen_native_forward=forward
 wrapper=ns.load_old('_spatial_runtime_full_TTA',R/'studies/map_paper_completion_v1/spatial_runtime_v2.py',common)
 try:wrapper.run('predict','vidstg');assert read(PREV/'spatial/vidstg_sealed.json')['queries']==10303;seal('vidstg')
 except BaseException:atomic(D/'vidstg_prediction_state.json',dict(status='failed_preserved',error=traceback.format_exc(),counts=e.actual));raise
 finally:runtime.load_source=old_load;native.frozen_native_forward=old_forward;e.close()
if __name__=='__main__':run()
