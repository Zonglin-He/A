"""Rerun the exact estimators with more training-only search evaluations."""
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from inputs import HOME, PRIVATE, RESULTS, atomic_json, csv_rows, load, read, sha
from rank import features, rank_loss, training_groups


FILES = ('inputs.py','rank.py','verify_winner.py','run.py','protocol.json')
CHECKED = ('SUMMARY.json','rank_rows.csv','verification_rows.csv')


def exercise(folder):
    for f in FILES:
        shutil.copyfile(HOME/f,folder/f)
    (folder/'results').mkdir()
    shutil.copyfile(RESULTS/'WINNER_AUDIT.json',folder/'results/WINNER_AUDIT.json')
    path = folder/'rank.py'
    code = path.read_text()
    assert code.count("'maxfev': 240") == 1
    path.write_text(code.replace("'maxfev': 240","'maxfev': 1200"))
    spec = read(folder/'protocol.json')
    assert '240 objective evaluations' in spec['rank_optimizer']
    spec['rank_optimizer'] = spec['rank_optimizer'].replace('240 objective evaluations', '1200 objective evaluations')
    (folder/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    with (PRIVATE/'numeric_budget.log').open('x') as log:
        subprocess.run([sys.executable,str(folder/'run.py')],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)


def scores(before, after):
    selected = ('D','legacy_pair','pair_entity','pool_same_frames','pool_all','pool_centered','pool_shuffled')
    metrics = {}
    for scope in before['rank']:
        metrics[scope] = {}
        for arm in selected:
            if arm in before['rank'][scope]:
                a,b = before['rank'][scope][arm],after['rank'][scope][arm]
                vals = ('covered','correct','instance_mrr','recovered','lost')
                metrics[scope][arm] = {v:[a[v],b[v]] for v in vals}
    return metrics


def training_improvement(old_fits, new_fits, data):
    by_key = {d.alias:d for d in data}
    decreases = []
    for f, g in zip(old_fits,new_fits,strict=True):
        assert f['held']==g['held'] and f['cases']==g['cases']
        assert f['calibration']==g['calibration'] and f['rank'].keys()==g['rank'].keys()
        train = [by_key[q] for q in f['calibration']]
        for arm, old in f['rank'].items():
            if old['optimizer'] in ('fixed','legacy','empty'):
                continue
            new = g['rank'][arm]
            matrices = [(x, group, features(x,arm)) for x,group in training_groups(train,arm)]
            objective = rank_loss(matrices,pair=arm=='pair_entity')
            delta = objective(np.asarray(old['weights'])) - objective(np.asarray(new['weights']))
            decreases.append(float(delta))
    return dict(comparisons=len(decreases),largest_decrease=float(max(decreases)),
                largest_increase=float(max(0.,-min(decreases))),median=float(np.median(decreases)),
                decreases_over_1e_4=sum(x>=1.e-4 for x in decreases),
                increases_over_1e_4=sum(x<=-1.e-4 for x in decreases))


def main():
    result = RESULTS/'NUMERICAL_CHECK.json'
    assert not result.exists()
    old = read(RESULTS/'SUMMARY.json')
    data, _, _ = load()
    keep = PRIVATE/'numeric_budget'
    keep.mkdir(mode=0o700)
    with TemporaryDirectory(dir=HOME.parent,prefix='.full_pool_numeric_') as root:
        temp = Path(root)
        exercise(temp)
        new = read(temp/'results/SUMMARY.json')
        old_fits = read(PRIVATE/'FIT_DETAILS.json')
        new_fits = read(temp/'private/FIT_DETAILS.json')
        decrease = training_improvement(old_fits,new_fits,data)
        changes = {}
        for file,arm in (('rank_rows.csv','arm'),('verification_rows.csv','rule')):
            origin = {(e['case'],e[arm]):e for e in csv_rows(RESULTS/file)}
            observed = {(e['case'],e[arm]):e for e in csv_rows(temp/'results'/file)}
            assert origin.keys()==observed.keys()
            cols=('winner_status','instance_rank') if arm=='arm' else ('winner_status','reject')
            changes[file]=dict(Counter(key[1] for key,e in origin.items()
                                       if any(e[c]!=observed[key][c] for c in cols)))
        for name in CHECKED:
            shutil.copyfile(temp/'results'/name,keep/name)
        for name in ('RUN_LOCK.json','FIT_DETAILS.json'):
            shutil.copyfile(temp/'private'/name,keep/name)
    stable = old['decision']==new['decision']
    new_verify = new['verification']['enriched_waymo']
    old_verify = old['verification']['enriched_waymo']
    keys=('correct_accepted','covered_wrong_rejected','natural_miss_rejected','winner_auc_covered')
    atomic_json(result,dict(bounds_per_start=[240,1200],scope='same data and objectives, no target input',
        old_status=old['ranking_optimizer_statuses'],new_status=new['ranking_optimizer_statuses'],
        objective_decrease=decrease,changes=changes,ranking_metrics_240_and_1200=scores(old,new),
        enriched_verification_240_and_1200={r:{k:[old_verify[r][k],new_verify[r][k]] for k in keys} for r in old_verify},
        decision_unchanged=stable,original_decision=old['decision'],larger_budget_decision=new['decision'],
        original_completed_summary=sha(RESULTS/'SUMMARY.json'),larger_budget_summary=sha(keep/'SUMMARY.json'),
        protected_data_access=False))
    print('Larger budget decision unchanged',stable,'objective decreases',decrease,flush=True)


if __name__=='__main__':
    main()
