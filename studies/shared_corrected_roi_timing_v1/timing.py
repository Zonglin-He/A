"""Actual uncached matched timing for direct shared-corrected-box ROI features."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'studies/shared_corrected_roi_v1'))
import shared_common as selected
from shared_common import *
import textwrap,argparse
S=Path(__file__).parent;D=R/'outputs/shared_corrected_roi_timing_v1';P=R/'results/shared_corrected_roi_timing_v1';G=D/'deployment'
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 selected.validate();return p
def current(ds,key):
 assert read(selected.D/f'components_{ds}_reporting_receipt.json')['status']=='verified_whole_domain';e=fixed(selected.D/f'components_{ds}_sealed.json')['files'][key];path=selected.D/e['path'];assert sha(path)==e['sha256']and sha(selected.D/e['receipt_path'])==e['receipt_sha256'];return path,e['sha256']
def prepare():
 assert not(D/'plan.json').exists();selected.validate();source=PREV/'timing_selection.json';atomic(D/'timing_selection.json',read(source));code=[Path(__file__),S/'controller.py',R/'stvg/current_shared_roi.py',R/'studies/map_paper_completion_v1/deployment.py',R/'studies/map_paper_completion_v1/timing_export_v1.py'];inputs=[selected.D/'plan.json',selected.D/'preflight_receipt.json',source,R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 p=dict(schema='shared_corrected_ROI_actual_uncached_timing_v1',authorization='Synchronize actual matched timing after user-specified first-auxiliary-DP removal, final DP retained.',science=dict(queries_per_domain=16,parents_per_domain=16,repeats=3,measured_Source_calls=192,measured_expert_calls=96,warmup_Source_calls=4,warmup_expert_calls=2,profile='shared_corrected_roi_v1',kappa=0,beta=1,first_aux_DP=False,final_DP=True,native224_FP32=True,same_expert8=True),timing='Full decode/parse/CPU-GPU transfers/Source/teacher request/geometry/direct E-ROI pool/prototype/jointMAP/final DP/cleanup in wall. Model startup and warmup separate; weights resident original offload; no cross-query RGB/features/teacher/prediction caches. All192outputs bitwise independently audited selected clean result. Nested subcomponents not additive; fixed16panel shared-host conditions, not full-domain throughput.',gate='Both selected clean whole-domain official score/allarray audits and initial owner successful actual exit. Pause only independently verified own CPU-sync process group, prevent new children by stopping controller too; finally resume on success or failure. No unrelated services signaled. One global GPU lease.',limits=dict(minimum_RAM=8*2**30,minimum_disk=80*2**30,private_bytes=160*2**30,stage_seconds=259200),failure='Preserve all attempts/partial measurements/logs/source/plan, no skip/fallback/hash waiver/blind timed call retry.',corruption_release_allowed=False,code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
 for f in code:
  dest=D/'source_snapshot'/f.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
 print(p['fingerprint'])
def run():
 validate();old=load_old('_shared_ROI_deployment_parent',R/'studies/map_paper_completion_v1/deployment.py',suite());src=textwrap.dedent(inspect.getsource(old.run));needle='from live_map import predictor_class,finish';assert src.count(needle)==1;src=src.replace(needle,'from stvg.current_shared_roi import predictor_class,finish').replace('geometry/auxDP/repool/one-step/MAP/finalDP/cleanup','geometry/direct shared-corrected-box ROI pool/one-step/MAP/finalDP/cleanup; first auxiliary DP absent')
 scope={**vars(old),'D':D,'P':P,'G':G,'validate':lambda _:validate(),'current_MAP':current};exec(compile(src,'<actual_shared_corrected_ROI_uncached_timing>','exec'),scope);scope['run']()
def export():
 validate();old=load_old('_shared_ROI_timing_export_parent',R/'studies/map_paper_completion_v1/timing_export_v1.py',suite());scope={**old.main.__globals__,'D':D,'P':P,'G':G,'Q':P/'deployment','validate':lambda _:validate(),'__file__':__file__};f=types.FunctionType(old.main.__code__,scope,'export_shared_ROI_timing');f()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','export']);globals()[p.parse_args().command]()
