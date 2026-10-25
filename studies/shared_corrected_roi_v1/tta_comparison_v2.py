"""Saved, audited full TTA scalars compared with the independently recomputed current method."""
from shared_common import *
import argparse
def run(ds):
 validate();ap=D/f'components_{ds}_reporting_receipt.json';audit=read(ap);assert audit['status']=='verified_whole_domain'
 td=R/'outputs/tta_full_v1';ta=read(td/f'{ds}_reporting_receipt.json');assert ta['status']=='verified_full_domain_Tent_SAR' and ta['all_native_joint_readouts_replayed_exact'] and ta['all_means_recomputed']
 tt={r['id']:r for r in read(td/ds/'metrics.json')['private_rows']};rr=[]
 for r in read(D/'components'/ds/'metrics.json')['private_rows']:
  t=tt[r['id']];assert t['parent']==r['parent']and t['duration_group']==r['duration_group']and t['values']['Source']==r['values']['Source']and t['values']['SourceJointMAP']==r['values']['SourceJointMAP']
  rr.append({**r,'values':{a:r['values'][a]for a in ['Source','SourceJointMAP','FullMAP']}|{a:t['values'][a]for a in ['TentNative','SARNative','TentJointMAP','SARJointMAP']}})
 aa=['Source','SourceJointMAP','TentNative','SARNative','TentJointMAP','SARJointMAP','FullMAP'];pp=[('FullMAP',a)for a in aa if a!='FullMAP'];folder=P/'matched_TTA'/ds
 report=suite().parent_report(rr,aa,pp,folder);suite().signed_tails(rr,pp,folder)
 from root_export_v1 import tails
 atomic(folder/'all_paired_tails.json',{a+'-'+b:tails(rr,a,b)for a,b in pp})
 (folder/'README.md').write_text('Full-domain matched comparison using already audited Tent/SAR logits/readouts and the independently recomputed shared-corrected-box ROI FullMAP. No optimizer, Source, expert or GT score is repeated. Source per-offset nativeMAP/hull and jointMAP readouts remain separate; native-to-joint gains are not optimizer gains. TTA is the pinned episodic five-step temporal-decoder LayerNorm-affine STVG adapter with fixed Source boxes, not canonical BN/full-network/continual methods. Its zero-expert/five-step budget differs from Full eight-expert/zero-step. Actual prior optimizer/acquisition costs are nonzero and retained in the original full-TTA results. All parent-cluster pointwise CIs, signed negatives, severe union/rescues and short/long splits are retained. Datasets were already exposed.\n')
 rec=dict(status='root_matched_saved_TTA_comparison_complete',dataset=ds,current_audit_sha256=sha(ap),TTA_audit_sha256=sha(td/f'{ds}_reporting_receipt.json'),current_metrics_sha256=sha(D/'components'/ds/'metrics.json'),TTA_metrics_sha256=sha(td/ds/'metrics.json'),report_sha256=sha(folder/'report.json'),code_sha256=sha(Path(__file__)),new_optimizer=0,new_Source=0,new_expert=0,new_GT_score=0);atomic(D/f'{ds}_TTA_comparison_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec);print(ds,'saved full TTA comparison synchronized')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);run(p.parse_args().dataset)
