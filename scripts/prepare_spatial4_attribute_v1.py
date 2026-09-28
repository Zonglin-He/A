"""F37 bounded preregistration; no label reads, no parent writes."""
import copy
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha

OUT=ROOT/'artifacts/decota_spatial4_attribute_v1'
F35=ROOT/'artifacts/decota_spatial10_components_v1'
F36=ROOT/'artifacts/decota_simplification_partial_v1'


def main():
    assert not (OUT/'LOCK.json').exists(),'already locked'
    p35,p36=read(F35/'LOCK.json'),read(F36/'LOCK.json')
    protected={}
    for base in (F35,F36):
        seal=read(base/'COMPLETION.json')
        protected.update(seal['files']);protected.update(seal['code_pins'])
        protected[str((base/'COMPLETION.json').relative_to(ROOT))]=sha(base/'COMPLETION.json')
    protected.update(p36['protected_pins'])
    for f,h in protected.items():assert sha(ROOT/f)==h,f
    rows=copy.deepcopy(p36['rows']);expanded={};core={};counts={}
    for c,rr in rows.items():
        seen={r['group'] for r in rr}
        expanded[c]=[copy.deepcopy(r) for r in p35['rows'][c] if 'core' in r['roles'] and r['group'] not in seen]
        core[c]=[r['key'] for r in p35['rows'][c] if 'core' in r['roles']]
        for r in rr:r['f37_role']=r['f36_role']
        for r in expanded[c]:r['f37_role']='space_extension'
        assert not {r['input']['video_sha256'] for r in rr}&{r['input']['video_sha256'] for r in expanded[c]}
        counts[c]=dict(main_queries=len(rr),dev_sources=8,eval_sources=16,space_extension=len(expanded[c]),core_sources=len(core[c]))
        dev=[r for r in rr if r['f37_role']=='development']
        for i,r in enumerate(dev):r['attribute_donor_key']=dev[(i+1)%len(dev)]['key']
        for r in rr:
            if r['f37_role']=='evaluation':r['attribute_donor_key']=dev[0]['key']
    lock=dict(name='DeCoTA_F37_spatial4_query_attribute',created_unix=time.time(),
        attachments={a:sha(Path('./private_authorization_notes')/a/'pasted-text.txt') for a in
          ['private-authorization-ce44067d7d36','private-authorization-0dafe5e1a3d6']},
        rows=rows,space_extension=expanded,core_keys=core,H_rows=p36['H_rows'],counts=counts,
        caps=dict(fits=600,backward=6000,new_DINO=256),protected_pins=protected,
        space=dict(candidate='exact F36 A4',steps=10,parameters=1792,lr={'hcstvg1_test':.01,'vidstg_test':.1},
            planned=4,gamma=0,kappa=None,accepted_weight=1,observation='same four physical original observations; original full-text S2',
            reuse='F35 raw expert cache and C0 parent; F36 A4 reused where exact; new fits only for65 extension sources',
            primary='S4-S0 on F36-excluded sources; F36 A4 results posthoc and separate; whole F35 core descriptive',
            noninferiority_pp=.5,no_A5=True,no_LR_search=True),
        temporal=dict(arms=['T0','T1','T2','T3','T4','T5'],parameters=66306,steps=5,lr=[1e-4,1e-3],beta=[.1,1.],
            prerequisites=['original checkpoint-compatible motion class mapping','weights exact','native attribute readout reproduction','visual responsiveness','nontrivial differentiable time path'],
            hard_block='Missing original class map or invalid actual input/gradient: document block; no random head, reconstructed semantics, or fake zero-gradient training.',
            objective='query affirmative target-motion positives only + beta KL(P_phi||P0); no coverage/no crop pseudo-boundaries',
            proposed_bias='finite log(max(r,1e-6)) added at existing visual attention score before spatial softmax; audit actual normalization first',
            spatial_condition='finite -2 outside Bprime at spatial token centers; unchanged full RGB and context; B0 control',
            native_readout='actual trained ASA logits followed by sigmoid once; never actionness/TTS',
            auxiliary_extension='No implicit flattening of time into spatial tokens or changed frame pooling hidden as native equivalence. Any actual nontrivial auxiliary implementation must be separately locked before eval.',
            selection='max source-macro fixed-Spatial10 v on8dev per direction; <=.1pp tie lower LR then higher beta; controls same configuration',
            conditional10='development only: if T5-T0 mean >=.005 and T2-T0 <=0, one10step check with same selected config; no eval-triggered search',
            fallback='unsupported retained native time; do not call unsupported/fallback-only a successful parameter method',
            GT_online=False),
        practical=dict(enabled_if_budget=True,supervision='original Spatial10 real weak-pair S2 anchors, not GT',
            scopes=[1792,2820],lr_multipliers=[.5,1.],steps=10,gamma=1e-4,regularizer_denominator=1792,
            kappa={'hcstvg1_test':None,'vidstg_test':2},
            selection='per direction and per scope equal 8dev/2LR opportunity; max source v; tie within .1pp lower multiplier',
            evaluation='same16eval per direction plus all6 fixed H cases separate; overlap not additive; no per-case switch',
            no_new_GT_diagnosis=True,no_scope_promotion=True),
        uncertainty=dict(seed=20260914,bootstrap=10000,neutral_pp=.1,historical_exposure=True,untouched=False),
        no_promotion=True,no_new_models=True,no_corruption=True,no_full_dataset=True,no_schedule=True)
    write(OUT/'LOCK.json',lock)
    write(OUT/'SOURCE_MANIFEST.json',dict(rows=rows,extension=expanded,H_rows=p36['H_rows'],counts=counts,
        exposure='All historical. Extension is F36-main-source-disjoint, not untouched.',overlap_not_additive=True))
    write(OUT/'STATUS.json',dict(stage='locked_interface_audit_and_S4',created=time.time(),finished=False))
    print(counts,flush=True)

if __name__=='__main__':main()
