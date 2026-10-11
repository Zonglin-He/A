"""Independent anonymous-row parent aggregation and paired bootstrap readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.analyze_stvg_opd_mechanism_review_v1 import BASE,PUB,QUALITY_FIELDS,PROP_FIELDS,GROUPS,verify

def rows(name):
    with gzip.open(PUB/(name+'.jsonl.gz'),'rt') as f:return [json.loads(x) for x in f]

def check_stat(raw,fields,expected,frame=False):
    cells=collections.defaultdict(list)
    for r in raw:cells[(r['dataset'],r['source_id'],r['condition'],r['order'])].append(r)
    by=collections.defaultdict(list)
    for key,rr in cells.items():
        if frame:by[key[:2]].append({f:np.mean([r[f] for r in rr]) for f in fields})
        else:by[key[:2]].extend(rr)
    assert len(raw)==expected['cells'] if not frame else len(cells)==expected['cells']
    count=0
    for field in fields:
        sample=np.array([np.mean([r[field] for r in rr if r.get(field) is not None]) for rr in by.values() if any(r.get(field) is not None for r in rr)])
        saved=expected['metrics'][field];assert len(sample)==saved['parents']
        if len(sample)==0:assert saved['mean'] is None and saved['ci95'] is None;continue
        assert abs(float(sample.mean())-saved['mean'])<2e-12
        # Independent construction and reduction; use identical registered parent RNG.
        rng=np.random.default_rng(20261011);boot=np.empty(10000)
        for lo in range(0,10000,100):
            ix=rng.integers(len(sample),size=(100,len(sample)));boot[lo:lo+100]=np.sum(np.take(sample,ix),axis=1)/len(sample)
        assert np.allclose(np.percentile(boot,[2.5,97.5]),saved['ci95'],rtol=0,atol=2e-12)
        assert int(np.count_nonzero(sample<-.05))==saved['parent_negative_tail_5pp'];count+=5
    return count

def run():
    verify();completion=read(BASE/'CPU_COMPLETION.json')
    for p,h in completion['files'].items():assert sha(ROOT/p)==h
    q=rows('QUALITY_ROWS');f=rows('FRAME_ROWS');c=rows('CELL_ROWS');s=read(PUB/'SUMMARY.json');count=0
    assert len(c)==1792 and len(q)==completion['quality_frame_cells'] and len(f)==completion['propagation_frame_cells']
    for r in q:
        assert r['quality']==('correct' if r['expert_GT_IoU']>=.5 else 'incorrect')
        assert r['reachability']==('zero' if r['max_first_reward']==0 else 'positive')
        assert r['Full_harm']==float(r['Full_current']<-.05) and r['Direct_harm']==float(r['Direct_current']<-.05)
        assert abs(r['Full_current']-r['Direct_current']-r['Full_minus_Direct_current'])<2e-12
        count+=5
    for r in f:
        assert r['group'] in GROUPS
        assert abs(r['After']-r['Before']-r['After_minus_Before'])<2e-12
        assert abs(r['After']-r['DINO']-r['After_minus_DINO'])<2e-12
        if r['group']=='outside_native':assert r['After_task_gated']==r['Native_task_gated']==0
        if not r['saved_decoder_support']:assert r['After']==r['Before']==r['Native']==r['DINO']==r['Direct']==0
        count+=4
    for key,items in s['quality'].items():
        setting,pop=key.rsplit('/',1)
        eligible=[r for r in q if r['setting']==setting and (pop=='all_historical' or not r['development'])]
        for stratum,item in items.items():
            qual,action=stratum.split('/');rr=[r for r in eligible if (qual=='all' or r['quality']==qual) and (action=='all' or r['reachability']==action)]
            assert len(rr)==item['expert_frame_cells'];count+=check_stat(rr,QUALITY_FIELDS,item['statistics'],True)
    for key,items in s['propagation'].items():
        setting,pop=key.rsplit('/',1);eligible=[r for r in f if r['setting']==setting and (pop=='all_historical' or not r['development'])]
        for group,item in items['all_eligible'].items():
            rr=[r for r in eligible if r['group']==group];count+=check_stat(rr,PROP_FIELDS,item['statistics'],True)
            assert len(rr)==item['frame_cells']
        observed={(r['dataset'],r['source_id']) for r in eligible if r['group']=='observed'}
        for group,item in items['common_parent_intersections'].items():
            parents=observed&{(r['dataset'],r['source_id']) for r in eligible if r['group']==group};assert len(parents)==item['common_parents']
            for label,g in [('observed','observed'),('comparison',group)]:
                rr=[r for r in eligible if r['group']==g and (r['dataset'],r['source_id']) in parents]
                count+=check_stat(rr,PROP_FIELDS,item[label],True)
    for key,items in s['oracles'].items():
        setting,pop=key.rsplit('/',1);rr=[r for r in c if r['setting']==setting and (pop=='all_historical' or not r['development'])]
        fields=list(items['pooled']['metrics']);count+=check_stat(rr,fields,items['pooled'])
        for order,item in items['per_order'].items():count+=check_stat([r for r in rr if r['order']==order],fields,item)
    trigger=s['pilot_resource_trigger'];actual=[]
    for item in trigger['settings']:
        rr=[r for r in q if r['setting']==item['setting'] and r['quality']=='correct'];zero=[r for r in rr if r['reachability']=='zero']
        parents=len({(r['dataset'],r['source_id']) for r in zero});ratio=len(zero)/len(rr) if rr else 0
        ci=s['quality'][item['setting']+'/all_historical']['correct/all']['statistics']['metrics']['Direct_minus_Full_current']['ci95']
        fire=parents>=10 and ratio>=.25 and ci is not None and ci[0]>0
        assert item['triggered']==fire and item['zero_correct_parents']==parents and abs(item['zero_correct_fraction']-ratio)<2e-12;actual.append(fire)
    assert trigger['status']==('triggered' if any(actual) else 'not_triggered')
    write(BASE/'ROOT_STATISTICS_READBACK.json',dict(status='pass',scope='independent anonymous frame/cell/parent aggregation and registered bootstrap, not full decoder Jacobian',
       matched_cells=len(c),expert_frame_cells=len(q),propagation_frame_cells=len(f),scalar_comparisons=count,bootstrap_draws=10000,
       pilot_trigger_status=trigger['status'],CPU_only=True,new_model_calls=0,GT_read='already_postseal_offline_scalar_records',
       summary_sha256=sha(PUB/'SUMMARY.json'),CPU_completion_sha256=sha(BASE/'CPU_COMPLETION.json'),time=time.time()))
    write(BASE/'REWARD_PILOT_DECISION.json',dict(status=trigger['status'],scope='fixed resource trigger only; no main-method promotion',
       main_method_changed=False,pilot_GPU_launched=False,new_model_calls=0,settings=trigger['settings'],root_readback_sha256=sha(BASE/'ROOT_STATISTICS_READBACK.json'),time=time.time()))
    print('MECHANISM_ROOT_PASS',count,trigger['status'],flush=True)

if __name__=='__main__':run()
