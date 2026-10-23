"""Stage exact executed cost source in a clean, already-sealed reproduction root."""
import sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'tools'))
from full import RUN,PROVIDER,read,sha,atomic,binding,check
S=Path(__file__).resolve().parent;D=R/'outputs/maintenance/simplepath_fair_cost_20261023'
parent=check(True);protocol=read(R/'results/simplepath_full_v2/cost_protocol.json');assert parent['fingerprint']==protocol['full_fingerprint'];assert all((RUN/ds/'score_receipt.json').exists()for ds in parent['jobs'])
assert not(D/'plan.json').exists(),'Never overwrite prior timing plan or repeat its budget; use a clean reproduction checkout'
D.mkdir(parents=True,exist_ok=True);shutil.copyfile(S/'worker.py',D/'run.py');assert sha(D/'run.py')==protocol['executed_source_sha256'];selection={}
for ds,entries in read(S/'selection_hashes.json').items():
 rows=[__import__('json').loads(line)for line in(PROVIDER/parent['jobs'][ds]['manifest']).read_text().splitlines()];ix={binding(row,parent['provider_runtime_fingerprint']):row for row in rows};selection[ds]=[ix[e['input_binding']]for e in entries]
 assert len(selection[ds])==16
atomic(D/'selection.json',selection);p={k:v for k,v in protocol.items()if k not in ['selection_hashes','executed_source','executed_source_sha256','raw_inputs_exported']};p.update(full_plan_sha256=sha(RUN/'plan.json'),script_sha256=sha(D/'run.py'),selection_sha256=sha(D/'selection.json'));atomic(D/'plan.json',p)
print('Prepared only; run private staged run.py once, with the shared GPU lock available.')
