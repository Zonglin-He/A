"""Write-once support construction; independent scalar cell/neighbour audit before GPU."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import *
import numpy as np
import torch
from vg_tta.desta3d_v3_context_mask import context_masks
DEST=OUT/'context_support_inputs_v1'

def independent_masks(correct,wrong,ids):
    result={}
    for b,ob in [('event','temporal'),('spatial','spatial')]:
        result[b]={}
        for which,old in [('correct',correct),('wrong',wrong[ob])]:
            x=old[b].numpy();y=x.copy()
            if b=='event':
                s,e=wrong['diagnostic']['temporal'][which+'_interval']
                before=[(fid,i) for i,fid in enumerate(ids) if fid<s]
                after=[(fid,i) for i,fid in enumerate(ids) if fid>=e]
                positions=([max(before)[1]] if before else [])+([min(after)[1]] if after else [])
                for i in positions:y[:,i]=1
            else:
                _,t,h,w=x.shape
                for k in range(t):
                    for r in range(h):
                        for c in range(w):
                            y[0,k,r,c]=max(x[0,k,rr,cc] for rr in range(max(0,r-1),min(h,r+2)) for cc in range(max(0,c-1),min(w,c+2)))
            result[b][which]=y
    return result

def main():
    assert not DEST.exists();torch.set_num_threads(4)
    rows=read(PANEL/'INPUTS.json');records=[];files={};maxerr=0.
    for i,row in enumerate(rows):
        old=OUT/'oracle001/episodes'/f'{i:02}'
        c=torch.load(old/'ORACLE_MASKS.pt',weights_only=False,map_location='cpu')
        w=torch.load(old/'WRONG_MASKS.pt',weights_only=False,map_location='cpu')
        v=context_masks(c,w,row['input']['frame_ids']);other=independent_masks(c,w,row['input']['frame_ids'])
        for b in other:
            for which in other[b]:
                err=float(np.max(np.abs(other[b][which]-v['masks'][b][which][b].numpy())));maxerr=max(maxerr,err);assert err==0
        p=DEST/f'{i:02}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(v,p);files[str(p)]=sha(p)
        records.append(dict(key=row['key'],source=row['source'],diagnostic=v['diagnostic'],old_files={str(p):sha(p) for p in [old/'ORACLE_MASKS.pt',old/'WRONG_MASKS.pt']}))
    write(DEST/'READBACK.json',dict(status='passed',queries=16,records=records,scalar_numpy_max_error=maxerr,
        source_GT_already_exposed=True,optimizer=0,GPU=False,target_read=False,
        counts={b:dict(original=sum(r['diagnostic'][b]['original_eligible'] for r in records),distinct=sum(r['diagnostic'][b]['support_distinct'] for r in records),matched_non_neutral=sum(r['diagnostic'][b]['eligible_changed_support'] for r in records),
            one_sided_neutral=sum(r['diagnostic'][b]['correct']['neutral']!=r['diagnostic'][b]['wrong']['neutral'] for r in records)) for b in ['temporal','spatial']}))
    write(DEST/'SEAL.json',dict(files=files,readback_sha=sha(DEST/'READBACK.json'),helper_sha=sha(ROOT/'vg_tta/desta3d_v3_context_mask.py')))
    print(read(DEST/'READBACK.json')['counts'])
if __name__=='__main__':main()
