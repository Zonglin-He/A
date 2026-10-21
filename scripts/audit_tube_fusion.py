"""Read-only posterior/identity audit of sealed HC64 cached fusion; no GT/head."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tube_fusion.run import TRIAL,read,sha,validate,atomic
from stvg.temporal import final_mbr

def main():
    lock=read(TRIAL/'lock.json');receipts=validate(lock);seal=read(TRIAL/'sealed.json')
    prior=read(ROOT/'outputs/tube_temporal_hc64/runtime_fingerprint.json')
    fingerprint=hashlib.sha256(json.dumps(prior,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert fingerprint==lock['old_lock']['runtime_fingerprint']
    assert prior['assets'][lock['checkpoint_path']]==lock['checkpoint_sha256']
    checks=0;changed_G=changed_P=0;heads=0
    for key in lock['ids']:
        r=receipts[key];q=torch.load(TRIAL/r['posterior_path'],map_location='cpu',weights_only=False)
        assert q['GT_read']is False
        with np.load(TRIAL/seal['files'][key]['path'],allow_pickle=False)as pred:
            for mode in('G','T','P'):
                layers=q['layers'][mode];assert layers.shape==(6,len(pred['frame_ids']),len(pred['frame_ids']))
                assert np.isfinite(layers).all()and(layers>=0).all()
                assert np.array_equal(layers,np.triu(layers,1))and np.allclose(layers.sum((1,2)),1,atol=1e-12,rtol=0)
                assert np.array_equal(q['means'][mode],np.mean(layers,axis=0));checks+=6
            assert np.array_equal(q['final']['Global6'],q['means']['G'])
            assert np.array_equal(q['final']['GlobalTube6'],.5*q['means']['G']+.5*q['means']['T'])
            assert np.array_equal(q['final']['GlobalPermuted6'],.5*q['means']['G']+.5*q['means']['P'])
            for method,table in q['final'].items():
                interval=final_mbr(pred['frame_ids'].tolist(),table)['interval'];assert np.array_equal(pred[method+'_interval'],interval)
            changed_G+=int(not np.array_equal(pred['Global6_interval'],pred['GlobalTube6_interval']))
            changed_P+=int(not np.array_equal(pred['GlobalPermuted6_interval'],pred['GlobalTube6_interval']))
        heads+=r['head_calls']
    assert checks==1152 and heads==384
    out=dict(status='all64_saved_posterior_identity_audit_passed',queries=64,GT_read=False,new_head_encoder_decoder_expert_calls=0,
        layers_normalized_then_averaged_checks=checks,exact_fixed_half_fusion=True,physical_MBR_recovered_queries=64,
        same_checkpoint_as_original_capture=True,checkpoint_sha256=lock['checkpoint_sha256'],original_runtime_fingerprint=fingerprint,
        final_layer_logits_recovery_batches=384,old_spatial_recovery_paths=192,actual_new_head_batches_in_trial=heads,
        actual_extra_RGB_encoder_decoder_expert_updates_in_trial=0,changed_intervals_fused_vs_global=changed_G,changed_intervals_fused_vs_permuted=changed_P,
        fingerprint=lock['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),seal_sha256=sha(TRIAL/'sealed.json'),
        interpretation='Saved-cache arithmetic and identity audit, not new prediction or task scoring. Strong is not an efficacy gate.')
    atomic(ROOT/'results/tube_fusion_hc64/structure.json',out);atomic(TRIAL/'structure_receipt.json',dict(public_sha256=sha(ROOT/'results/tube_fusion_hc64/structure.json'),audit_source_sha256=sha(__file__),GT_read=False))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
