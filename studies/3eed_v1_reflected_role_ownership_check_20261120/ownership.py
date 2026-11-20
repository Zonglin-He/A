"""Score frozen physical hypotheses under the two already-published instance-association rules."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'3eed_v1_reflected_relation_profiles_20261120'
sys.path.insert(0,str(PREVIOUS))
from observe import WORK, ROOT, digest, read, save
from predict_profiles import DECISIONS,SIGNATURE
from profiles import ARMS
from coverage_profiles import associations
from score_profiles import (COHORTS,PANELS,SOURCES,anon,historical_endpoints,historical_records,inventory,sealed_inputs,state)
from diagnose_roles import old_seed_owners

RESULTS=HERE/'results'


def outcomes(pool_s,pool_r,enabled,s_id,r_id):
    s=s_id in pool_s if s_id is not None else None
    r=bool(enabled and r_id in pool_r) if r_id is not None else None
    return dict(has_subject=s,has_reference=r,both=(s and r) if None not in (s,r) else None,
                only_subject=(s and not r) if None not in (s,r) else None,
                only_reference=(r and not s) if None not in (s,r) else None,
                neither=(not s and not r) if None not in (s,r) else None)


def run(repo):
    sealed_inputs()
    assert digest(SIGNATURE)==read(HERE/'PROTOCOL.json')['decision_seal']
    history=historical_records(repo)
    strict=pd.read_csv(PREVIOUS/'results/coverage_rows.csv.gz',keep_default_na=False).set_index(['group','case','source','pool'])
    previous=pd.read_csv(PREVIOUS/'results/all_rows.csv.gz',keep_default_na=False).set_index(['group','case','source','arm'])
    covers=[]
    selected=[]
    shape_count=0
    for panel,cohort in COHORTS.items():
        for case,entry in enumerate(read(PANELS[panel])):
            row=entry['row'];target=row['platform']
            labels,_=inventory(row['metapath'],target,int(row['suffix'] or 0),int(row['object_index']))
            ends,_=historical_endpoints(cohort,entry,labels,history)
            for source in SOURCES:
                if source==target: continue
                tag=f'{panel}_{case:03d}_{source}.npz'
                old=f"{cohort}/{entry['case']:02d}_{source}.npz"
                original=ROOT/'crossplatform/round171/predictions'/old
                assert digest(original)==history[0][old]
                case_token=anon(f'{panel}|{case}')
                meta=dict(panel=panel,case=case_token,source=source,target=target,dual_known=int(None not in ends))
                with np.load(original,allow_pickle=False) as historical,np.load(WORK/tag,allow_pickle=False) as z,np.load(DECISIONS/tag,allow_pickle=False) as d:
                    # Score each unique shape once, irrespective of the arm or role which nominated it.
                    slots=z['subject'][:2].reshape(-1,7);refs=z['reference'][:2].reshape(-1,7)
                    native=z['pool'][:2].reshape(-1,7)
                    selected_shapes=[d[k] for k in ('raw','refs','boxes')]
                    chunks=[slots,refs,native,*selected_shapes]
                    stack=np.vstack(chunks)
                    valid=np.isfinite(stack).all(axis=1)
                    distinct,inverse=np.unique(stack[valid],axis=0,return_inverse=True)
                    independent=old_seed_owners(distinct,historical['seeds'],labels)
                    strict_owners=associations(distinct,labels)
                    shape_count+=len(distinct)
                    for rule,physical in [('unique_iou',strict_owners),('seed_ownership',independent)]:
                        owners=np.full(len(stack),-1,dtype=int)
                        owners[valid]=physical[inverse]
                        so,ro,whole,before,reference,final=np.split(owners,np.cumsum([len(c) for c in chunks])[:-1])
                        for name,s,r,valid_r in [('original_roles',so[:4],ro[:4],bool(z['original_ref_valid'])),
                                                ('coherent_roles',so,ro,bool(z['original_ref_valid'])),
                                                ('original_pca_pool',whole[:256],whole[:256],True),
                                                ('coherent_pca_pools',whole,whole,True)]:
                            row_counts=outcomes(s,r,valid_r,*ends)
                            if rule=='unique_iou' and None not in ends:
                                saved=strict.loc[panel,case_token,source,name]
                                assert all(int(saved[k])==int(v) for k,v in row_counts.items())
                            covers.append(dict(**meta,active=int(z['active']),rule=rule,pool=name,**row_counts))
                        for i,arm in enumerate(ARMS):
                            truth_s=state(before[i],ends[0]); truth_r=state(reference[i],ends[1])
                            old_row=previous.loc[panel,case_token,source,arm]
                            if rule=='unique_iou':
                                assert (truth_s,truth_r,state(final[i],ends[0]))==(old_row.human_raw,old_row.reference,old_row.human_subject)
                            known=None not in ends
                            selected.append(dict(**meta,active=int(z['active']),rule=rule,arm=arm,
                                raw_subject=truth_s,reference=truth_r,final_subject=state(final[i],ends[0]),
                                both_raw=(truth_s==truth_r=='T') if known else None,
                                both_final=(state(final[i],ends[0])==truth_r=='T') if known else None,
                                raw_iou=old_row.raw_iou,iou=old_row.iou,best_all_union=old_row.best_all_union))
    RESULTS.mkdir(exist_ok=True)
    c=pd.DataFrame(covers); s=pd.DataFrame(selected)
    assert len(c)==168*2*4 and len(s)==168*len(ARMS)*2
    c.to_csv(RESULTS/'coverage_rows.csv.gz',index=False)
    s.to_csv(RESULTS/'prediction_rows.csv.gz',index=False)
    group_counts=[]; role_counts=[]
    for panel in ('natural170','natural171','natural_all'):
        cc=c if panel=='natural_all' else c[c.panel==panel]
        ss=s if panel=='natural_all' else s[s.panel==panel]
        for (rule,pool),g in cc[cc.dual_known==1].groupby(['rule','pool']):
            for scope,block in [('all',g),('active',g[g.active==1])]:
                counts={k:int(block[k].sum()) for k in ('both','only_subject','only_reference','neither')}
                assert sum(counts.values())==len(block)
                group_counts.append(dict(panel=panel,rule=rule,pool=pool,scope=scope,n=len(block),**counts))
        for (rule,arm),g in ss.groupby(['rule','arm']):
            same=g.raw_subject=='T'
            new_hit=g.iou>.50; old_hit=g.raw_iou>.50
            role_counts.append(dict(panel=panel,rule=rule,arm=arm,n=len(g),known_both=int(g.dual_known.sum()),
               both_raw=int(g.both_raw.sum()),both_final=int(g.both_final.sum()),
               correct_reference=int((g.reference=='T').sum()),correct_raw_subject=int(same.sum()),
               known_subject=int((g.raw_subject!='NA').sum()),known_reference=int((g.reference!='NA').sum()),
               correct_final_subject=int((g.final_subject=='T').sum()),
               shape_rescue50=int((same&new_hit&~old_hit).sum()),shape_harm50=int((same&~new_hit&old_hit).sum()),
               beyond_all=int((same&new_hit&(g.final_subject=='T')&(g.best_all_union<=.50)).sum())))
    pd.DataFrame(group_counts).to_csv(RESULTS/'coverage.csv',index=False)
    pd.DataFrame(role_counts).to_csv(RESULTS/'identities.csv',index=False)
    direction_counts=[]
    comparisons=[]
    for (source,target),block in s.groupby(['source','target']):
        for (rule,arm),part in block.groupby(['rule','arm']):
            d=part.set_index(['panel','case']).sort_index()
            baseline=block[(block.rule==rule)&(block.arm=='v1')].set_index(['panel','case']).sort_index()
            assert d.index.equals(baseline.index)
            known=d.dual_known==1
            positive=d[known].both_raw.astype(bool)
            old=baseline[known].both_raw.astype(bool)
            direction_counts.append(dict(source=source,target=target,rule=rule,arm=arm,n=len(d),dual_known=int(known.sum()),
                dual_correct=int(positive.sum()),dual_gain_vs_v1=int((positive&~old).sum()),
                dual_loss_vs_v1=int((old&~positive).sum())))
    pc=c[c.dual_known==1]
    for (source,target,rule,pool),d in pc.groupby(['source','target','rule','pool']):
        comparisons.append(dict(source=source,target=target,rule=rule,pool=pool,n=len(d),
            **{k:int(d[k].sum()) for k in ('both','only_subject','only_reference','neither')}))
    pd.DataFrame(direction_counts).to_csv(RESULTS/'direction_identities.csv',index=False)
    pd.DataFrame(comparisons).to_csv(RESULTS/'direction_coverage.csv',index=False)
    save(RESULTS/'audit.json',dict(n=168,arms=len(ARMS),rules=2,shape_count=shape_count,
        prediction_rows=len(s),coverage_rows=len(c),independent_dual_n=42,prior_decision_hash=digest(SIGNATURE),
        historical_commit='fa61a76d10e31790e2be031a77126814b8010643',all_unique_iou_arms_reproduced=True,
        FINAL_access=False,source_training=False,uses_labels_only_after_predictions=True,
        output_hashes={f.name:digest(f) for f in sorted(RESULTS.iterdir()) if f.name!='audit.json'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    run(p.parse_args().archive)

