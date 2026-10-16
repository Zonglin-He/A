"""Finite-protocol closing/accounting from sealed evidence only, no new GT."""
from pathlib import Path
import json,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy
read,sha=c.read,c.sha
def put(p,d):return c.write(p,d,replace=True)
def main():
    c.verify();b=c.BASE;groups=('global_B','confirmation128','ablation_noCaption','ablation_noCoverage','tubedetr_qualification8');publications=[]
    for group in groups:
        p=b/'publication'/group;d=read(p/'FINAL_GITHUB_RECEIPT.json');assert d['status']=='pass' and read(p/'PUBLICATION_COMPLETION.json')['status']=='complete';publications.append(dict(group=group,**d))
    ledger=read(b/'BUDGET_LEDGER.json');reserved={k:sum(r[k] for r in ledger['reservations']) for k in ledger['caps']};actual=[]
    for group,base in [('original_G32_A',c.OLDGLOBAL),('global_B',b/'global_B'),('confirmation_A',b/'confirmation128/arms/A'),('confirmation_B',b/'confirmation128/arms/B'),('noCaption',b/'ablation_noCaption'),('noCoverage',b/'ablation_noCoverage')]:
        teacher=read(base/'teacher/G32/TEACHER_GLOBAL_BARRIER.json');qual=read(base/'NATIVE_QUALIFICATION.json');n=qual.get('additional_generation_calls',3*qual.get('qualified_domains',0));assert n==6
        actual.append(dict(group=group,WHO_actual=teacher['new_generation_calls'],qualification_generation_calls=n,response_receipts=teacher['response_count']))
    fits=465+sum(r['actual_fit'] for r in read(c.OLDGLOBAL/'POSTSEAL_RESULTS.json')['usage'])+8
    for scope in ('global_B','confirmation128','ablation_noCaption','ablation_noCoverage'):fits+=sum(r['actual_fit'] for r in read(b/scope/'POSTSEAL_RESULTS.json')['usage'])
    temporal=read(b/'confirmation128/temporal/TEACHER_GLOBAL_BARRIER.json')['new_caption_alignment_pairs'];totals=dict(WHO=sum(r['WHO_actual']+r['qualification_generation_calls'] for r in actual),Temporal=temporal,fits=fits)
    assert totals==dict(WHO=1440,Temporal=436,fits=822)
    assert all(totals[k]<=reserved[k]<=ledger['caps'][k] for k in totals)
    account=dict(status='pass',count=128,caps=ledger['caps'],reservations_preserved=reserved,actual=totals,WHO_breakdown=actual,Spatial_fit_breakdown=dict(A_BT_and_B_BT=465,original_G32_A=31,global_B=25,confirmation_A=128,confirmation_B=108,noCaption=32,noCoverage=25,TubeDETR_engineering_steps=8),qualification_generations_are_not_full96token_WHO=True,Temporal_unit='caption-alignment pairs, not individual generation calls',exact_reuse_and_no_fit_not_new_fit=True,GT_read=False)
    put(b/'ROOT_BUDGET_READBACK002.json',account)
    decision=dict(status='complete',count=128,scope='finite overnight measurements and scientific decision only; not whole-project FINAL',formal_outcome='unresolved_spatial_gain',interpretation='B adds Spatial value over Temporal but fails matched A advantage in both domains',method_selection_stopped=True,no_new_search=True,TubeDETR_native_qualification='pass_interface_only',TubeDETR_B_effects='not_triggered_confirmation_failed',fullset='not_started_not_authorized',CURRENT_unchanged=True,semantic512_blind_panel='pending, not substituted by task scores or old16 labels',pending_items=['512-window identity/role/full-extent arm-blind semantic review; not claimed completed','historically uninstrumented latency subcomponents remain unknown'],scientific_model_work_remaining_in_night=0,GT_read=False)
    put(b/'ROOT_FINITE_DECISION002.json',decision)
    out=ROOT/'results/stvg_overnight_consolidation_20261016/2026-10-17/closing';out.mkdir(exist_ok=True,parents=True)
    put(out/'BUDGET_READBACK.json',account);put(out/'FINITE_DECISION.json',decision);put(out/'VERIFIED_PUBLICATIONS.json',dict(status='pass',records=publications))
    cost=read(b/'cost/SAVED_MATCHED32_COST.json');assert read(b/'cost/CPU_COMPLETION.json')['result_sha256']==sha(b/'cost/SAVED_MATCHED32_COST.json');put(out/'SAVED_MATCHED32_COST.json',cost)
    manifest=read(b/'TABLE1_MANIFEST_BINDING.json');put(out/'TABLE1_MANIFEST_STATUS.json',dict(status='prepared_only',original_query_count=13785,HC_queries=3482,Vid_queries=10303,scientific_B_runtime_not_registered=True,fullset_started=False,GT_read=False))
    for name in ('report_stvg_overnight_closing002.py','publish_stvg_overnight_group002.py','report_stvg_overnight_saved_cost_v1.py'):
        p=ROOT/'scripts'/name;t=out/'code'/name;t.parent.mkdir(exist_ok=True);t.write_text(public_source_text(p.read_bytes()))
    text='''# Finite overnight consolidation: measured outcomes and limits

The six core256 development groups, both global32 designs, the independent128-parent A/B confirmation, two prelocked32 deletions and native TubeDETR8 interface probes have completed their authorized measurements. No model work remains in this finite night plan. This does not mean whole-project FINAL: the prelocked512-window semantic panel remains unlabelled and pending. No semantic identity/role/completeness conclusion is inferred from task IoU, feature edges or old16 labels.

| Confirmation128 (64/domain) | HC vIoU % | VidSTG vIoU % |
|---|---:|---:|
| Frozen |22.702652|10.420386|
| Temporal |28.035312|10.970497|
| A |28.662751|11.648711|
| Frozen candidate B-fullstep |28.492622|11.394287|

B minus Temporal is +0.457311/+0.423790pp; B minus matched A is -0.170129/-0.254425pp. Thus Spatial still contributes, but B's proposed extra evidence advantage did not confirm. Stop method selection; do not promote either method by reselecting on confirmation, add a gate, tune parameters/Prompt or start official fullset. No claim of statistical significance or general impossibility. Sources exclude core256 and16 registered historical cohorts, not proof of exhaustive globally unseen history.

On fixed development32, deleting the whole-caption selector gives +0.152285/-1.894123pp vs LocalB; this includes28 newly eligible WHO calls on7 missing-caption sources. Common-caption HC13/Vid12 gives -0.054972/-2.553279pp. Deleting only candidate-priority coverage gives +0.002029/+0.578743pp (common-caption +0.002498/+0.771657pp). Actual all128-window indices/quotas/protection were independently recomputed for each deletion. These are diagnostic deletions, not a new selected recipe after failed confirmation.

The released domain-specific TubeDETR weights passed all8 engineering interface probes, with512 last-decoder norm3 parameters, native stride2/one offset, exact cached/full outputs and rollback. The synthetic one-step self-box+0.01 proposal is not the B optimizer. Scientific B transfer effect comparisons remain zero because confirmation advancement failed. No new model download.

Budget: actual WHO generations1404 plus36 short qualification generations =1440 <=2560 cap; Temporal436 caption-alignment pairs <=512; Spatial822 fits/engineering steps <=1280, including465 tonight A_BT/B_BT fits,31 original G32-A fits,25 globalB,236 confirmation,57 deletion fits,8 engineering steps. Conservative reservations remain1572/512/857 and were not erased. Source128 new captures and warm restores are separate from Spatial fits; reused/no-fit cases are not extra parents or free cold-deployment work.

Saved matched32 Full/A/B costs retain Source capture/restore, full readout and Teacher prefill/generation instrumentation. Cold captures, warm restores and no-fit reuse differ. Historical candidate-preparation/token-selection/solver subdivisions that were not instrumented remain unknown. No newly proven15% end-to-end saving. Table1 original manifests bind13785 queries (HC3482/Vid10303) but no new B fullset runtime or results. CURRENT, Bootstrap and online carry remain unchanged.

All complete scientific groups preserve positive/negative/neutral/unknown outcomes and were sealed before GT. Each group has its own verified publication receipt. Actual plot pixels, privacy, HTTP bytes/SHA256/Gitblob, fetched modes/trees and both main/research refs were checked. Public exports exclude media, annotations/GT coordinates, weights, raw caches/tensors/optimizer states, actual query-caption prompts, Teacher raw responses, conversations and credentials.
'''
    (b/'NEXT_ACTION.md').write_text('# Exact next action after finite night measurements\n\nNo further scientific model stage in this night plan: B advancement failed. Complete root publication/archive closure only. Keep all negative/partial/recovery artifacts. Do not revive queues or select a replacement from confirmation. TubeDETR B transfer/fullset not triggered; CURRENT unchanged.\n\nPending beyond measured evidence: prelocked512-window arm-blind semantic labels and missing historical latency subdivisions. Do not claim these completed or infer labels from outcomes. No new Prompt/parameter/frame search.\n')
    (b/'ALL_GROUPS_SUMMARY.md').write_text(text);(b/'TABLE1_STATUS.md').write_text('# Table1 status\n\nNOT STARTED. Original13785-query manifests prepared only. B did not confirm over A; no automatic B fullset or scientific TubeDETR effects. CURRENT unchanged.\n')
    doc=ROOT/'docs/STVG_NIGHT_FINITE_CLOSING002.md';doc.write_text(text)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));x=list(range(2));ax.bar([v-.17 for v in x],[.4573107867,.4237896242],width=.34,label='B minus Temporal');ax.bar([v+.17 for v in x],[-.1701288747,-.2544245661],width=.34,label='B minus A');ax.set(xticks=x,xticklabels=['HC64','Vid64'],ylabel='Confirmation Student vIoU difference (pp)',title='Spatial gain remains; B extra value over A did not confirm');ax.axhline(0,color='black',lw=.7);ax.legend();fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig)
    paths={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT))};verify_privacy(paths)
    # Only public anonymous aliases may appear in saved cost/source lists.
    for rel in paths:
        if (ROOT/rel).suffix=='.json':
            txt=(ROOT/rel).read_text()
            import re
            assert not re.search(r'"(?:HC[0-9]{5}|V[0-9]{5})"',txt),rel
    pub=b/'publication/closing';put(pub/'EXPORT_MANIFEST.json',dict(status='complete',count=128,paths=sorted(paths),files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in sorted(paths)]));print(json.dumps(dict(status='exported_pending_root_public_closing',files=len(paths),actual_budget=totals)))
if __name__=='__main__':main()
