"""CPU import-path repair for official NestedTensor pickles; scoring unchanged."""
import os,sys,runpy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'external/TA-STVG'))
import utils.misc,utils.video_list
from research.native_event_text_v1.run import D,read,put,sha
assert read(D/'sealed.json')['count']==64
put(D/'score_import_recovery.json',dict(reason='Official NestedTensor class module absent from CPU sys.path',preserved_failure='score.log',new_adapter_sha256=sha(__file__),original_score_sha256=sha(Path(__file__).parent/'score.py'),inference_changes=0,scoring_math_changes=0,GT_read_only_after_seal=True))
runpy.run_path(str(Path(__file__).parent/'score.py'),run_name='__main__')
