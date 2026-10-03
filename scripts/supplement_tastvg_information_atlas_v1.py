"""Order readbacks from sealed anonymous metrics; no new readout/fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_information_atlas_math_v1 import bootstrap_summary

def run(folder):
    for ds in ['vidstg','hc2']:
        rows=json.loads((folder/ds/'ROWS.json').read_text());ans={}
        for mode in ['clean','corrupt']:
            for panel in ['all','search','confirm']:
                rr=[r for r in rows if r['domain']=='target' and (r['condition']=='clean')==(mode=='clean') and (panel=='all' or r['panel']==panel)]
                for order in sorted({r['order'] for r in rr}):
                    z=[r for r in rr if r['order']==order];name=f'{mode}/{panel}/{order}'
                    ans[name]=dict(coverage=dict(cells=len(z),sources=len({r['source_index'] for r in z})),
                        metrics={k:bootstrap_summary(z,k) for k in z[0]['metrics']})
        (folder/ds/'ORDER_DIAGNOSTICS.json').write_text(json.dumps(ans,indent=2,allow_nan=False)+'\n')
        print('ATLAS_ORDER_READBACK',ds,list(ans),flush=True)

if __name__=='__main__':run(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_temporal_information_atlas/2026-10-03')
