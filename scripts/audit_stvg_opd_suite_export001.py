"""Portable fixed-suite closing contracts and public byte bindings, without private data."""
import hashlib
import json
import sys
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(folder):
    folder=Path(folder);inv=read(folder/'PHASE_CLOSURE_INVENTORY.json');count=0
    assert inv['phase_count']==6 and [p['phase'] for p in inv['phases']]==['P1','P2','P3','P4','P5','P6']
    assert inv['original_P0_negative_gate_preserved'] and not inv['current_method_changed'] and not inv['active_original_suite_model_controllers']
    assert inv['original_current_method_sha256']==inv['actual_current_method_sha256']
    assert inv['EATA_and_historical_queues_paused'] and inv['no_result_driven_tuning'] and inv['new_GPU_calls']==0
    sizes=[41355,7168,7168,15504,2560,64]
    for i,item in enumerate(inv['phases'],1):
        label=item['phase'];close=folder/'receipts'/(label+'_ROOT_CLOSING_RECEIPT.json');c=read(close)
        assert c['status']=='complete' and not c['paper_suite_complete'] and label in c['scope']
        assert item['root_closing_sha256']==digest(close) and item['phase_actual_complete'] and item['all_negative_results_retained']
        assert item['logical_arrivals']==sizes[i-1]
        gpath=folder/'receipts'/(label+'_FINAL_GITHUB_RECEIPT.json');g=read(gpath)
        assert g['status']=='pass' and g['repository']=='Zonglin-He/A' and g['commit']==item['original_public_commit']==c['public_commit']
        assert item['complete_original_remote_verification_receipt_sha256']==digest(gpath)
        assert len(g['files'])==g['file_count']==item['original_public_files']
        assert sum(f['bytes'] for f in g['files'])==g['bytes']==item['original_public_bytes']
        if i==1:
            assert g['per_remote_file_content_verified'] and set(g['refs'].values())=={g['commit']}
            assert all(f['remote_bytes_verified'] and f['remote_blob_verified'] for f in g['files'])
        elif i==2:
            assert g['all_remote_contents_individually_verified']
            assert all(f['remote_bytes_SHA256_gitblob_tree_and_local_exact'] for f in g['files'])
        else:
            assert g['actual_remote_bytes_SHA256_Git_blob_tree_verified']
            assert all(all(f[k] for k in ['remote_bytes_exact','remote_SHA256_exact','remote_Git_blob_exact','remote_tree_mode_exact']) for f in g['files'])
        for name,key in [('ARCHIVE_VERIFICATION_RECEIPT','actual_archive_receipt_sha256'),('ROOT_VISUAL_REVIEW','actual_view_receipt_sha256'),('PUBLIC_SCALAR_AUDIT','public_scalar_audit_sha256')]:
            p=folder/'receipts'/(label+'_'+name+'.json');assert read(p)['status']=='pass' and digest(p)==item[key]
        assert digest(folder/'phase_reports'/('STVG_OPD_REVISED_'+label+'_ROOT_REVIEW.md'))==item['report_sha256']
        count+=len(g['files'])*4+15
    assert inv['phases'][3]['logical_arrivals']==969*16
    assert inv['phases'][5]['deployment_outputs']==64*4
    for f in inv['figures']:
        root=folder/'receipts'/f['namespace']/'ROOT_CLOSING_RECEIPT.json';g=folder/'receipts'/f['namespace']/'FINAL_GITHUB_RECEIPT.json'
        assert read(root)['status']=='complete' and read(g)['status']=='pass'
        assert digest(root)==f['root_sha256'] and digest(g)==f['public_receipt_sha256'];count+=4
    c2=read(folder/'receipts/P2_ROOT_CLOSING_RECEIPT.json');c3=read(folder/'receipts/P3_ROOT_CLOSING_RECEIPT.json');c5=read(folder/'receipts/P5_ROOT_CLOSING_RECEIPT.json');c6=read(folder/'receipts/P6_ROOT_CLOSING_RECEIPT.json')
    assert not c2['Full_minus_Direct_superiority_established']
    assert not c3['uniform_Full_minus_LN_only_superiority_established'] and not c3['uniform_Full_minus_alpha0_superiority_established']
    assert c5['K4_minus_K2_and_K8_minus_K4_Vid_CI_contain_zero']
    assert not c6['offline_GT_head_fair_deployment_baseline'] and not c6['head_uniform_superiority_to_direct_projection_established']
    bind=read(folder/'CODE_BINDING.json')
    for rel,r in bind['public_files'].items():
        p=folder/rel;assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'];count+=2
    return dict(status='pass',scope='portable original fixed six phase closure/negative control/public verification metadata and exact final companion bytes',
        comparisons=count,phase_count=6,original_immutable_phase_bindings=inv['immutable_phase_closing_bindings_checked'],
        no_private_GT_media_fit_reconstruction=True,no_new_GPU_or_scoring=True,no_retuning_or_promotion=True,
        all_negative_findings_preserved=True,paper_suite_complete=False)


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),sort_keys=True))
