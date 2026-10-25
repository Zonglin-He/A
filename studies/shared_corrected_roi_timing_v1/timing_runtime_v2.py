"""Per-call resource enforcement outside the matched query timer; unchanged science."""
from timing import *
import shutil
def execution_validate():
 p=validate();e=read(D/'execution_v2_plan.json');assert e['science_plan_sha256']==sha(D/'plan.json')and digest({k:v for k,v in e.items()if k!='fingerprint'})==e['fingerprint']
 for f,h in e['code_sha256'].items():assert sha(R/f)==h,f
 return p
def run():
 p=execution_validate();started=time.perf_counter()
 def resource_guard():
  assert time.perf_counter()-started<p['limits']['stage_seconds'],'72h stage limit; preserve measurements'
  assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=p['limits']['minimum_RAM'],'RAM guard; preserve measurements'
  assert shutil.disk_usage(R).free>=p['limits']['minimum_disk'],'Disk guard; preserve measurements'
  assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['limits']['private_bytes'],'Private storage guard; preserve measurements'
 resource_guard();old=load_old('_shared_ROI_guarded_deployment_parent',R/'studies/map_paper_completion_v1/deployment.py',suite());src=textwrap.dedent(inspect.getsource(old.run));needle='from live_map import predictor_class,finish';assert src.count(needle)==1;src=src.replace(needle,'from stvg.current_shared_roi import predictor_class,finish');needle="attempt=G/'attempts'/";assert src.count(needle)==1;src=src.replace(needle,"resource_guard();attempt=G/'attempts'/");src=src.replace('geometry/auxDP/repool/one-step/MAP/finalDP/cleanup','geometry/direct shared-corrected-box ROI pool/one-step/MAP/finalDP/cleanup; first auxiliary DP absent')
 scope={**vars(old),'D':D,'P':P,'G':G,'validate':lambda _:execution_validate(),'current_MAP':current,'resource_guard':resource_guard};exec(compile(src,'<resource_guarded_actual_shared_corrected_ROI_timing_v2>','exec'),scope);scope['run']()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['run']);run()
