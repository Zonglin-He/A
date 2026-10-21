"""Engineering checks only; never select parameters or run a dataset experiment."""
import sys,argparse,json,hashlib,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',choices=['hc2','vidstg']);a=p.parse_args()
 if not a.smoke:
  subprocess.run([sys.executable,'-B','-m','pytest','-q','tests'],cwd=ROOT,check=True);return
 import numpy as np
 ds=a.smoke;request=ROOT/f'data/evaluation/smoke_{ds}.json';r=json.loads(request.read_text());out=ROOT/f'outputs/checks/{ds}.npz';tick=time.time();subprocess.run([sys.executable,'-B','-m','stvg','predict','--request',str(request),'--output',str(out)],cwd=ROOT,check=True)
 reference=ROOT/f"outputs/reference/predictions/{r['key']}.npz"
 with np.load(out)as new,np.load(reference)as old:
  checked=[]
  for key in ['Source','E_Ref','frame_ids','Native_interval','Strong_interval']:
   assert np.array_equal(new[key],old[key]),(ds,key,float(np.max(abs(new[key]-old[key]))));checked.append(key)
 receipt=dict(status='pass',dataset=ds,previously_evaluated_query=True,checked_arrays=checked,all_bitexact=True,new_data_or_parameter_selection=False,scope='native/source/ERef/Strong equivalence only; complete BoxFixed HC64 migration is separately recorded',GT_read=False,seconds=time.time()-tick)
 (out.with_suffix('.check.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
