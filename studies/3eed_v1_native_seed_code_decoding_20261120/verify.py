"""Independent replay from all positive/negative focal terms, with full row and pair accounting."""
import ast
import csv
import subprocess
import gzip
from collections import Counter
from hashlib import sha256

import numpy as np
from scipy.special import expit

from predict import HERE, NATIVE, OUT, PARENT, PROTOCOL, atomic, digest, pinned, prior, read
from seed_code import ARMS,FIELDS

PUBLIC=HERE/'results'
PRIVATE=HERE/'run_PRIVATE/postseal'
GUARDED='native_focal_abstain'


def table(name):
    with (PUBLIC/f'source_{name}.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def replay(z):
    x=z['logits'].astype(float)
    bce_yes=np.maximum(x,0)-x+np.log1p(np.exp(-np.abs(x)))
    bce_no=np.maximum(x,0)+np.log1p(np.exp(-np.abs(x)))
    p=expit(x)
    lp=.25*(1-p)**2*bce_yes
    ln=.75*p*p*bce_no
    raw=z['codes']
    inside=z['inside']
    top=np.zeros_like(raw)
    maxs=[]
    for m,row in enumerate(inside):
        order=sorted(np.flatnonzero(row),key=lambda j:(-x[j],j))
        top[m,order[:4]]=True
        maxs.append(x[order[0]] if order else -np.inf)
    np.testing.assert_array_equal(top,z['top'])
    np.testing.assert_array_equal(top.sum(1),raw.sum(1))
    def neg_energy(mask):
        return ln.sum()-np.stack([np.where(code,lp,ln) for code in mask]).sum(1)
    joint=z['joint'].astype(float)
    birth=x[z['birth']]
    verify_fields=dict(frozen_joint=joint,birth=birth,box_max=np.array(maxs),
                       native_logit=raw@x,native_focal=neg_energy(raw),top_logit=top@x,
                       top_focal=neg_energy(top),mismatch=neg_energy(raw[z['permutation']]))
    for field,measure in verify_fields.items():
        np.testing.assert_allclose(measure,z['fields'][FIELDS.index(field)],rtol=1e-11,atol=1e-12)
    pref=np.argsort(-joint,kind='stable')
    def best_measure(score):
        hi=score.max()
        if np.isneginf(hi):
            return pref[0]
        return next(j for j in pref if hi-score[j]<=1e-12)
    out={}
    for arm in ARMS:
        if arm=='v1':
            continue
        name=arm.removesuffix('_abstain')
        measure=verify_fields[name]
        out[arm]=pref[0] if arm.endswith('_abstain') and measure.max()<=1e-12 else best_measure(measure)
        assert out[arm]==z['choices'][ARMS.index(arm)],arm
    return out


def audit():
    revision='843a901f5367a6be5cc484b97c6ecad462961dfa'
    def released(path):
        return subprocess.check_output(['git','-C',str(NATIVE),'show',revision+':'+path])
    def encoder_node(text):
        return next(ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text))
                    if isinstance(n,ast.FunctionDef) and n.name=='_get_3eed_target_boxes')
    for name in ('models/losses.py','utils/box_util.py'):
        assert released(name)==(NATIVE/name).read_bytes()
    dataloader='src/joint_det_dataset.py'
    assert encoder_node(released(dataloader))==encoder_node((NATIVE/dataloader).read_bytes())
    original,roster=pinned()
    seal=read(OUT/'SEALED.json')
    truthseal=read(PRIVATE/'SEALED.json')
    assert (truthseal['predictions']==digest(OUT/'SEALED.json') and
            seal['n']==len(seal['files'])==192 and seal['labels_read'] is False)
    with gzip.open(PUBLIC/'source_all_rows.csv.gz','rt') as f:logged=list(csv.DictReader(f))
    anonymize=lambda p,i:sha256(f'seed-code-v1|public|{p}|{i}'.encode()).hexdigest()
    anon=sorted(((p,i) for p in roster for i in roster[p]),key=lambda a:anonymize(*a))
    assert len(anon)==len(logged)==192
    raw={}
    for platform,indices in roster.items():
        checked=prior.sealed_arrays(prior.R181,'SEALED.json','source_scores',platform,['ious'])['ious']
        file=PRIVATE/(platform+'.npz')
        assert digest(file)==truthseal['files'][file.name]
        with np.load(file,allow_pickle=False) as bank:
            np.testing.assert_allclose(bank['ious'],checked[indices],rtol=0,atol=1e-5)
            for k,i in enumerate(indices):
                rel=f'{platform}/{i:05}.npz'
                assert digest(OUT/rel)==seal['files'][rel] and digest(PARENT/rel)==original['files'][rel]
                with np.load(OUT/rel,allow_pickle=False) as z, np.load(PARENT/rel,allow_pickle=False) as old:
                    choices=replay(z)
                    choices['v1']=int(old['choices'][0])
                    np.testing.assert_array_equal(old['choices'][:3],z['choices'][:3])
                    np.testing.assert_array_equal(old['boxes'],z['boxes'])
                    slot=np.asarray([choices[arm] for arm in ARMS])
                    ious=checked[i]
                    gt=bank['gt_codes'][k]
                    native_chars=z['codes']
                    unique,idx,inv=np.unique(native_chars,axis=0,return_index=True,return_inverse=True)
                    representatives=[ix[np.argmax(z['joint'][ix])] for c in range(len(unique))
                                     if len(ix:=np.flatnonzero(inv==c))]
                    gate=bool(z['fields'][FIELDS.index('native_focal')].max()>0)
                    raw[platform,i]=(slot,ious[slot],ious,ious[representatives],np.all(native_chars==gt,axis=1),gate)
    for n,((platform,i),logged_row) in enumerate(zip(anon,logged),1):
        assert logged_row['row_id']==f'S{n:03}' and logged_row['platform']==platform
        slot,vec,ious,rep,matches,gate=raw[platform,i]
        assert (logged_row['active']=='True')==gate
        for k,arm in enumerate(ARMS):
            np.testing.assert_allclose(float(logged_row[arm+'_iou']),vec[k],rtol=0,atol=1e-5)
            assert int(logged_row[arm+'_switch'])==int(slot[k]!=slot[ARMS.index('frozen_joint')])
        assert (logged_row['gt_chosen']=='True')==bool(matches[slot[ARMS.index(GUARDED)]])
    def groups(name):
        return [val for (p,_),val in raw.items() if p==name or name=='equal']
    for row in table('metrics'):
        group=groups(row['group'])
        a=ARMS.index(row['arm'])
        v=np.asarray([x[1][a] for x in group])
        assert int(row['n'])==len(v) and int(row['hits25'])==sum(v>.25) and int(row['hits50'])==sum(v>.50)
        np.testing.assert_allclose(float(row['mIoU']),v.mean()*100,atol=1e-6)
    for row in table('paired')+table('fallback'):
        a,b=map(ARMS.index,(row['arm'],row['reference']))
        g=groups(row['group'])
        if state:=row.get('state'):
            def kept(x):
                base=ARMS.index('frozen_joint')
                switched=x[0][ARMS.index(GUARDED)]!=x[0][base]
                return {'abstain':not x[5],'active_same':x[5] and not switched,'active_switch':switched,'all':True}[state]
            g=list(filter(kept,g))
        assert int(row['n'])==len(g)
        for t in (25,50):
            assert int(row[f'rescue{t}'])==sum(x[1][a]>t/100>=x[1][b] for x in g)
            assert int(row[f'harm{t}'])==sum(x[1][b]>t/100>=x[1][a] for x in g)
        assert int(row['slot_switch'])==sum(x[0][a]!=x[0][b] for x in g)
    for row in table('identifiability'):
        g=groups(row['group'])
        cut=int(row['threshold'])/100
        assert int(row['any_class_representative_hit'])==sum(x[3].max()>cut for x in g)
        assert int(row['gt_code_present'])==sum(x[4].any() for x in g)
        assert int(row['frozen_miss_recoverable'])==sum(x[2].max()>cut>=x[1][ARMS.index('frozen_joint')] for x in g)
    cases=[r for r in logged if r['switched']=='True']
    harms=[r for r in cases if float(r['frozen_joint_iou'])>.5>=float(r[GUARDED+'_iou'])]
    details=dict(actual_switches=len(cases),harms50=len(harms),harm_gt_code_present=sum(r['gt_in_pool']=='True' for r in harms),
                 harm_chose_exact_gt_code=sum(r['gt_chosen']=='True' for r in harms),
                 harm_exact_gt_code_had_good_representative=sum(r['gt_representative_good50']=='True' for r in harms),
                 harmed_rows_with_any_mixed_code=sum(r['shared_codes50']=='True' for r in harms),
                 frozen_missed_but_recoverable_by_gate=dict(Counter('abstain' if r['active']=='False' else
                     ('switch' if r['switched']=='True' else 'keep') for r in logged if r['recoverable50']=='True')))
    saved=read(PUBLIC/'source_decision.json')
    assert saved['active_switch_rescues50']==sum(float(r[GUARDED+'_iou'])>.5>=float(r['frozen_joint_iou']) for r in cases)
    assert saved['active_switch_harms50']==len(harms) and saved['active_switch_harms50']>saved['active_switch_rescues50']
    assert saved['source_pass'] is False and details['harms50']>0
    files=[p for p in PUBLIC.iterdir() if p.is_file() and p.name!='audit.json']
    atomic(PUBLIC/'audit.json',dict(verified_records=len(raw),verified_arms=len(ARMS),verified_original_candidates=192*16,
           upstream_native_loss_and_membership_byte_exact=True,upstream_native_loader_method_ast_exact=True,
           full_focal_all_1024_replay=True,all_decisions_reproduced=True,all_public_row_ious_reproduced=True,
           all_paired_tables_reproduced=True,negative_cases=details,decision_verified=True,
           extra_confirmation_rows=0,target_records_evaluated=0,FINAL_access=False,
           result_sha256={p.name:digest(p) for p in sorted(files)}))
    print('Independent verification complete',len(raw),len(ARMS),details)


if __name__=='__main__':
    audit()
