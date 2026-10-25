"""Independent actual uncached kappa0 timing, same fixed16 parents x3 repeats."""
from ns_common import *
import textwrap

def run():
 validate();old=load_old('_legacy_timing',R/'studies/map_paper_completion_v1/deployment.py',suite());src=textwrap.dedent(inspect.getsource(old.run));assert src.count('from live_map import predictor_class,finish')==1;src=src.replace('from live_map import predictor_class,finish','from stvg.current import predictor_class,finish');assert src.count("z[arm]")==1 and src.count("z[arm+'_interval']")==1;src=src.replace('z[arm]',"z['NoShrink'if arm=='FullMAP'else arm]").replace("z[arm+'_interval']","z['NoShrink_interval'if arm=='FullMAP'else arm+'_interval']");scope={**vars(old),'D':D,'P':P,'G':D/'deployment','validate':lambda _:validate(),'current_MAP':no_shrink_path};exec(compile(src,'<actual_selected_kappa0_timing>','exec'),scope);scope['run']()
def no_shrink_path(ds,key):
 p,e=no_shrink(ds,key);return p,e['sha256']
def export():
 validate();old=load_old('_legacy_timing_export',R/'studies/map_paper_completion_v1/timing_export_v1.py',suite());scope={**old.main.__globals__,'D':D,'P':P,'G':D/'deployment','Q':P/'deployment','validate':lambda _:validate()};f=types.FunctionType(old.main.__code__,scope,'export_no_shrink_timing');f()
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('command',choices=['run','export']);z=a.parse_args();globals()[z.command]()
