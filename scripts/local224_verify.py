"""Read-only independent array/utility/pause verification after local224 seals."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.local224.run import OUT,PUBLIC,validate,read,write,sha,NAMES
from stvg.temporal import expected_iou_fast,probabilities

def main():
    torch.set_num_threads(1);p=validate();seal=read(OUT/'sealed.json');report=read(PUBLIC/'report.json')
    assert seal['queries']==64 and report['queries']==64;same=whole=post_checks=0;utility_checks=0;maxerror=0.
    for item,receipt in zip(p['items'],seal['files']):
        assert item['query']==receipt['query'];assert sha(OUT/receipt['prediction_path'])==receipt['prediction_sha256'];assert sha(OUT/receipt['posterior_path'])==receipt['posterior_sha256']
        v=torch.load(OUT/receipt['posterior_path'],map_location='cpu',weights_only=False);cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False)
        z=np.load(OUT/receipt['prediction_path']);ids=z['frame_ids'];g=v['qs']['Global6']
        if item['crop_stats']['windows_equal']:
            assert torch.equal(v['local']['Local224']['logits'],v['local']['Shifted224']['logits']);assert np.array_equal(z['GlobalPlusLocal224'],z['GlobalPlusShifted224']);same+=1
        if item['crop_stats']['area_fraction']==1:
            assert torch.equal(v['local']['Local224']['logits'],cap['temporal_cache']['offsets'][0]['native_logits']);assert np.array_equal(v['qs']['GlobalPlusLocal224'],v['qs']['GlobalPlusFull224']);whole+=1
        for label in ['Local224','Shifted224']:
            vv=v['local'][label];rec=dict(frame_ids=vv['ids']);grid=probabilities([rec,rec],[vv['logits'],vv['logits']])[0];q=np.zeros_like(g);mapping={int(f):i for i,f in enumerate(ids)}
            for (a,b),w in zip(grid['ij'].T,grid['p']):q[mapping[grid['ids'][a]],mapping[grid['ids'][b]]]+=w
            assert np.array_equal(v['qs']['GlobalPlus'+label],.5*g+.5*q);post_checks+=1
        # Brute-force finite IoU expectation at each method decision, including zero-mass decisions.
        q=v['qs']['GlobalPlusLocal224'];u=expected_iou_fast(ids,q);i,j=np.nonzero(q);starts=ids[i];ends=ids[j]+1
        for arm in ['Global6','GlobalPlusFull224','GlobalPlusLocal224','GlobalPlusShifted224']:
            a,b=z[arm+'_interval'];inter=np.maximum(0,np.minimum(b,ends)-np.maximum(a,starts));direct=float((q[i,j]*inter/((b-a)+(ends-starts)-inter)).sum());ai=int(np.where(ids==a)[0][0]);bi=int(np.where(ids==b-1)[0][0]);maxerror=max(maxerror,abs(direct-u[ai,bi]));utility_checks+=1
    assert same==whole==55 and post_checks==128 and utility_checks==256 and maxerror<2e-12
    full=read(ROOT/'outputs/boxfixed_full_v1/state.json');assert full['status']=='paused_by_user' and full['paused_completed']==6088
    pause=read(ROOT/full['pause_receipt']);assert len(pause['files'])==6088
    for key,rr in pause['files'].items():
        assert sha(ROOT/'outputs/boxfixed_full_v1/vidstg/predictions'/f'{key}.npz')==rr['prediction_sha256'];assert sha(ROOT/'outputs/boxfixed_full_v1/vidstg/receipts'/f'{key}.json')==rr['receipt_sha256']
    assert len(list((ROOT/'outputs/boxfixed_full_v1/vidstg/receipts').glob('*.json')))==6088 and not torch.cuda.is_initialized()
    result=dict(status='pass',frozen_full_sources_unchanged=434,identical_window_bitwise_checks=same,full_window_vs_old_native_logits_bitwise_checks=whole,independent_posteriors=post_checks,independent_utility_checks=utility_checks,utility_max_error=maxerror,paused_full_predictions_and_receipts_unchanged=6088,full_remains_paused_by_user=True,no_new_labels_read=True,CUDA_initialized=False)
    write(PUBLIC/'verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
