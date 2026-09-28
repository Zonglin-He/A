"""Fresh API equivalence on four exposed F45 cases; no new configuration selection."""
import argparse
import fcntl
import gc
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status

OUT=ROOT/'artifacts/decota_final_freeze_v1'
F45=ROOT/'artifacts/decota_structured_shrinkage_v1'
KEYS=['hcstvg1_test:000928','hcstvg1_test:000747','vidstg_test:001608','vidstg_test:009637']


def main():
    import torch
    from methods.decota_final_v1 import FinalDeCoTAPredictor
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from scripts.structured_temporal_shrinkage_v1 import cache_for,f44_rows
    p=read(F45/'LOCK.json');old=f44_rows();rows={r['key']:r for rr in p['rows'].values() for r in rr}
    lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    all_receipts=[]
    for c,direction in [('hcstvg1_test','vid_to_hc1'),('vidstg_test','hc2_to_vid')]:
        predictor=FinalDeCoTAPredictor.from_pretrained(direction)
        for key in [k for k in KEYS if k.startswith(c+':')]:
            dest=OUT/'api_equivalence'/(key.replace(':','_')+'.pt')
            if dest.with_suffix('.json').exists():
                receipt=read(dest.with_suffix('.json'));assert sha(dest)==receipt['sha256']
                all_receipts.append(receipt);continue
            assert not dest.exists()
            r=rows[key];cache=cache_for(r);frames,ids=decode(r['input'])
            tick=time.perf_counter();result=predictor.predict(frames,ids,r['input'])
            saved_a4=load(cache['A4_reference']['path'])['fits'][cache['A4_reference']['arm']]
            assert sha(cache['A4_reference']['path'])==cache['A4_reference']['sha256']
            assert result['spatial']['anchors']==saved_a4['anchors']
            assert result['spatial']['best_step']==saved_a4['best_step']
            assert all(torch.equal(v,saved_a4['state'][n]) for n,v in result['spatial']['state'].items())
            assert all(a['loss']==b['loss'] for a,b in zip(result['spatial']['path'],saved_a4['path']))
            old_fit=load(old[key]['fit_references']['P1']['path'])
            assert result['temporal']['best_step']==old_fit['best_step']
            assert all(torch.equal(v,old_fit['state'][n]) for n,v in result['temporal']['state'].items())
            assert all(a['loss']==b['loss'] for a,b in zip(result['temporal']['path'],old_fit['path']))
            prior=load(F45/('A_'+r['f45_role'])/(key.replace(':','_')+'.pt'))
            expected=prior['points']['P1']['0.25']
            assert all(torch.equal(v,expected['state'][n]) for n,v in result['anchored_temporal']['state'].items())
            assert torch.equal(result['boxes'],expected['parameter']['boxes'])
            assert result['indices']==expected['parameter']['indices']
            assert all(torch.equal(a,b) for a,b in zip(result['predictions']['Final_DeCoTA']['logits'],expected['parameter']['logits']))
            assert all(torch.equal(a['raw_logits'],b['raw_logits']) for a,b in zip(result['teacher'],cache['teacher']))
            save(dest,result)
            receipt=dict(key=key,path=str(dest),sha256=sha(dest),completed=time.time(),seconds=time.perf_counter()-tick,
                exact_A4_anchors_state_loss=True,exact_F44_hard_state_loss=True,exact_F45_eta025=True,
                exact_source_actionness=True,full_original_model_forward=result['full_forward_audit'],
                source_restored=result['source_restored'],GT_online=False,new_DINO=result['expert']['new_DINO'],
                cost=result['cost'])
            write(dest.with_suffix('.json'),receipt);all_receipts.append(receipt)
            print('F46 API EXACT',key,round(receipt['seconds'],2),flush=True)
            status(OUT/'STATUS.json',dict(stage='api_equivalence',done=len(all_receipts),total=4,finished=False))
            del result,frames;gc.collect();torch.cuda.empty_cache()
        del predictor;gc.collect();torch.cuda.empty_cache()
    write(OUT/'API_VERIFICATION.json',dict(status='pass',sources=4,receipts=all_receipts,
        new_DINO=sum(r['new_DINO'] for r in all_receipts),GT_online=False,configuration_selected=False,
        scope='exposed F45 cases; packaging equivalence, not new generalization evidence'))


if __name__=='__main__':main()
