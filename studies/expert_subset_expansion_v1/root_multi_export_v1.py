"""Read only fully audited scores; no expert/Source/GT or target selection."""
from common_v1 import *
from full_score import cluster

def main():
 validate();guard();torch.set_num_threads(1);summary={};text='# Fixed128 expert replacement: all four experts\n\nSame original128distinctparents/domain and eight original expert RGB frames/physical indices. One referring expert per branch, Source and current parameters fixed; no new Source/original ReferDINO/training. ReferFormer-R101 is a same-family capacity variant; MTTR is a distinct mask-output model with a declared mask-to-box/eight-frame adapter. Unequal pretraining/parameters/resolution/compute remain. Current uses unchanged soft observed candidates, not a hard-anchor correction.\n\n|Dataset|Expert|Pure@Source t / v %|Current own t / v %|R.3 / R.5 %|Unobserved IoU %|\n|---|---|---|---|---|---|\n'
 for ds in ['hc2','vidstg']:
  allrows={};audits={}
  for expert in REUSE+EXPERTS:
   folder=D/expert/ds;ack=read(folder/'reporting_receipt.json');assert ack['metrics_sha256']==sha(folder/'metrics.json')and ack['report_sha256']==sha(P/expert/ds/'report.json');allrows[expert]=read(folder/'metrics.json')['private_rows'];audits[expert]=dict(audit_sha256=sha(folder/'reporting_receipt.json'),report_sha256=ack['report_sha256']);m=read(P/expert/ds/'report.json')['means'];p=m['PureSource'];c=m['Current'];text+=f"|{ds}|{expert}|{100*p['t']:.6f} / {100*p['v']:.6f}|{100*c['t']:.6f} / {100*c['v']:.6f}|{100*c['vIoU_ge_03']:.6f} / {100*c['vIoU_ge_05']:.6f}|{100*c['unobserved_actual_sIoU']:.6f}|\n"
  base={r['id']:r for r in allrows['referdino']};effects={}
  for expert in ['referformer_r50',*EXPERTS]:
   join=[];diff=[]
   for r in allrows[expert]:
    b=base[r['id']];assert r['parent']==b['parent']and r['duration_group']==b['duration_group'];join.append(dict(parent=r['parent'],values=dict(Alternative=r['values']['Current'],Original=b['values']['Current'])))
    a={};z={}
    for f in r['values']['Current']:
     vals=[r['values'][arm][f]for arm in ['Current','PureSource']]+[b['values'][arm][f]for arm in ['Current','PureSource']]
     a[f]=None if any(v is None for v in vals)else(vals[0]-vals[1])-(vals[2]-vals[3]);z[f]=None if a[f]is None else 0.
    diff.append(dict(parent=r['parent'],values=dict(GainDifference=a,Zero=z)))
   fields=[(f,100)for f in ['t','v','unobserved_actual_sIoU']];effects[expert]=dict(Current_vs_ReferDINO=cluster(join,'Alternative','Original',fields),within_expert_gain_difference_vs_ReferDINO=cluster(diff,'GainDifference','Zero',fields))
  summary[ds]=dict(paired_expert_effects=effects,audits=audits)
 text+='\nEach own-expert Current-minus-PureSource and fixed-own-event Current-minus-PureOwn comparison, all signed severe/rescue outcomes, short/long parent confidence intervals and actual expert/cached costs are in the per-expert reports. The newly computed pure baselines have nearest-expert tails; previous E-tail hybrid values remain preserved separately, not renamed. Source/previous Current scores are reused, not rescored. New-model masks, E geometry, actual ROI features and own posterior/interval/final DP were recomputed. No expert/setting chosen using target scores. All confidence intervals are pointwise paired-parent10000 bootstrap on exposed panels, not independent unknown-test validation or proof of expert independence.\n'
 (P/'ANALYSIS.md').write_text(text);atomic(P/'between_expert_paired_effects.json',summary);atomic(D/'root_multi_export_receipt.json',dict(status='root_all_expert_paired_effects_and_audit_provenance_exported_PENDING_publication',summary_sha256=sha(P/'between_expert_paired_effects.json'),analysis_sha256=sha(P/'ANALYSIS.md'),code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_GT=0));print('root all expert scalar comparisons exported',flush=True)
if __name__=='__main__':main()
