"""Two existing-query GPU interface checks, no labels or parameter selection."""
import sys,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from stvg.runtime import setup,lease,sha
from stvg.tube_decode import decode
from stvg.tubedetr import load_source
from stvg.tensors import state_hash
from tools.boxfixed_tube.adapter import capture
from stvg.tube_runner import write


def main():
    setup();lock=lease();cfg=json.loads((ROOT/'configs/default.json').read_text());tube=json.loads((ROOT/'configs/tubedetr.json').read_text());rows=[]
    for ds,name,ref in [('vidstg','vid','hc2_to_vid'),('hc2','hc2','vid_to_hc2')]:
        row=json.loads((ROOT/f'data/evaluation/tubedetr_{name}_dev128.jsonl').read_text().splitlines()[0]);config={**cfg['datasets'][ds],**tube['common'],**tube['datasets'][ds]};model=load_source(config);before=state_hash(model.state_dict());frames,fids=decode(row['metadata'],ds);tick=time.perf_counter();cap=capture(model,frames,fids,row['metadata']['caption']);root=ROOT/f'outputs/tubedetr/{ref}';seal=json.loads((root/'sealed.json').read_text());e=seal['files'][row['id']];assert sha(root/e['path'])==e['sha256']
        with np.load(root/e['path'])as z:assert np.array_equal(cap['Source'].numpy(),z['Source'])and np.array_equal(cap['Native_interval'],z['Native_interval'])and np.array_equal(fids,z['frame_ids'])
        assert state_hash(model.state_dict())==before;rows.append(dict(dataset=ds,previously_evaluated_query=True,Source_Native_bitexact=True,postfusion_shape=list(cap['features'].shape),grid_shape=list(cap['grid_shape']),Global6_interval=cap['Global6_interval'],six_layers=True,no448=True,no_optimizer=True,GT_read=False,seconds=time.perf_counter()-tick));del model,cap,frames;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'results/tubedetr_boxfixed_development/interface_qualification.json',dict(status='pass',checks=rows,new_experiment_or_parameter_selection=False));lock.close()

if __name__=='__main__':main()
