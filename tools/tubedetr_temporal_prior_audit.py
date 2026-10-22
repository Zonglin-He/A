"""Post-hoc no-logit uniform-posterior MBR control, never selected for deployment."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from stvg.temporal import final_mbr
from stvg.runtime import sha
from stvg.tube_runner import write
from tools.tubedetr_temporal_leakage_audit import tiou
PUB=ROOT/'results/tubedetr_temporal_leakage_audit';OUT=ROOT/'outputs/tubedetr_temporal_leakage_audit'
p=json.loads((ROOT/'results/tubedetr_boxfixed_development/plan.json').read_text())
if __name__=='__main__':
    assert not(PUB/'uniform_prior_plan.json').exists()
    write(PUB/'uniform_prior_plan.json',dict(scope='Post-hoc leakage explanation, after whole-input baseline audit; no new method, tuning, labels online or full inference',hypothesis='High entropy endpoint posterior plus expected-IoU decoding may primarily induce a long central interval prior',control='Exact same physical MBR and legal endpoint set with uniform mass on all legal pairs; no caption, pixels, learned weights or logits. Zero hyperparameter search.',queries=12205,neural_calls=0,parent_offline_audit_sha256=sha(PUB/'offline_audit.json'),code_sha256=sha(Path(__file__))))
    result={};cache={};tick=time.time()
    for ds,d in p['datasets'].items():
        rs=[json.loads(x)for x in(ROOT/d['full_manifest']).read_text().splitlines()];truth={x['id']:x for x in map(json.loads,(ROOT/d['full_labels']).read_text().splitlines())};previous={x['id']:x for x in json.loads((OUT/ds/'details_private.json').read_text())};values=[]
        for i,r in enumerate(rs):
            ids=np.asarray(r['metadata']['frame_ids']);key=tuple(ids-ids[0])
            if key not in cache:
                q=np.triu(np.ones((len(ids),len(ids))),1);q/=q.sum();cache[key]=final_mbr(key,q)['interval']
            interval=(np.asarray(cache[key])+ids[0]).tolist();g=truth[r['id']]['interval'];values.append(dict(id=r['id'],uniform_prior=tiou(interval,g),interval=interval))
            if i%500==0:print(ds,i,'unique_grids',len(cache),flush=True)
        groups={}
        for name,vs in [('all',values),('dev128',[x for x in values if previous[x['id']]['development']]),('other_queries',[x for x in values if not previous[x['id']]['development']])]:
            groups[name]=dict(queries=len(vs),mean_tIoU_percent=100*float(np.mean([x['uniform_prior']for x in vs])),Global6_minus_uniform_prior_pp=100*float(np.mean([previous[x['id']]['global6']-x['uniform_prior']for x in vs])),uniform_prior_beats_Global6_queries=sum(x['uniform_prior']>previous[x['id']]['global6']+1e-12 for x in vs))
        result[ds]=groups;write(OUT/ds/'uniform_prior_details_private.json',values)
    write(PUB/'uniform_prior_audit.json',dict(status='complete',datasets=result,unique_relative_grids=len(cache),seconds=time.time()-tick,zero_learned_model_or_text_information=True,not_a_new_method_or_independent_test=True))
