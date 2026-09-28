"""Label-separated paired utility and independent chain/metric audit."""
import argparse
import collections
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.c1_controlled_corruption_v1 import *
from scripts.analyze_spatial_reference_absorption_v1 import score
from scripts.audit_c1_correction_mechanism_v1 import metric
from scripts.audit_c1_concept_state_diagnostic_v1 import overlap_np
METRICS=['vIoU_corrected','sIoU','tIoU','unobserved_sIoU','common_unobserved_sIoU']


def aggregate(values,rows):
    grouped=collections.defaultdict(list)
    for v,r in zip(values,rows):
        if v is not None:assert math.isfinite(v);grouped[r['source']].append(v)
    if not grouped:return dict(mean=None,ci95=None,sources=0)
    a=np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    rng=np.random.default_rng(20260923);boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),sources=len(a),
                median=float(np.median(a)),negative_gt5pp=int((a<-.05).sum()),minimum=float(a.min()),maximum=float(a.max()))


def diff(a,b):return a-b if a is not None and b is not None else None


def summarize(rows):
    arms=list(rows[0]['arms'])
    comparisons=[('C1','Frozen'),('Episodic','Frozen'),('Direct','Frozen'),('C1','Episodic'),('C1','Direct'),
                 ('C1_Before','Frozen_formal'),('C1','C1_Before'),('C1_fixed10','C1'),('Episodic','Frozen_formal')]
    return dict(n=len(rows),arms={a:{m:aggregate([r['arms'][a][m] for r in rows],rows) for m in METRICS} for a in arms},
        comparisons={a+'-'+b:{m:aggregate([diff(r['arms'][a][m],r['arms'][b][m]) for r in rows],rows) for m in METRICS} for a,b in comparisons},
        tails={a+'-'+b:{str(t):[r['key'] for r in rows if r['arms'][a]['vIoU_corrected']-r['arms'][b]['vIoU_corrected']<-t] for t in [.05,.1]} for a,b in comparisons},
        expert=dict(observations=sum(r['observations'] for r in rows),accepted=sum(r['anchors'] for r in rows),
            no_expert=sum(r['observations']==0 for r in rows),no_anchor=sum(r['anchors']==0 for r in rows),
            GT_valid_observations=sum(r['expert_quality']['valid_observations'] for r in rows),
            accepted_geometric_bad=sum(r['expert_quality']['accepted_IoU_lt03'] for r in rows),
            rejected_geometric_good=sum(r['expert_quality']['rejected_best_IoU_ge05'] for r in rows)))


def evaluate(stage):
    import torch
    from methods.decota_refine_uniform_v1.api import reconstruct
    p=verify();bar=read(OUT/stage/'BARRIER.json');assert sha(p['labels'])==p['labels_sha'];labels=read(p['labels'])
    rows=[];counts=collections.Counter();maximum=0.;previous={};prevhash={};common=collections.defaultdict(set)
    for f,h in read(OUT/'homogeneous/BARRIER.json')['files'].items():
        x=load(f);common[(x['model'],x['key'])].update(x['observed'])
    for f,h in bar['files'].items():
        assert sha(f)==h;receipt=read(Path(f).with_suffix('.json'));assert receipt['lock_sha']==sha(OUT/'LOCK.json')
        x=load(f);gt=labels[x['key']];key=(x['model'],x['stream']);fit=x['online'];ep=x['episodic'];tt=x['temporal']
        assert not x['GT_used'] and not x['corruption_label_used_by_method']
        assert x['previous_sha']==prevhash.get(key);prevhash[key]=h
        init=previous.get(key,x['source_state'])
        assert all(torch.equal(v,init[k]) for k,v in fit['initial_state'].items())
        for n,v in x['committed'].items():
            expect=torch.zeros_like(v) if n=='spatial.query_residual' else fit['initial_state'][n]+(fit['state'][n]-fit['initial_state'][n])/16
            assert torch.equal(v,expect);counts['committed_tensors']+=1
        previous[key]=x['committed'];counts['history_arrivals']+=1
        for z in [fit,ep]:
            assert z['failure'] is None and z['selected_step']==min(range(len(z['losses'])),key=lambda i:z['losses'][i])
            assert torch.equal(z['final']['boxes'],z['path'][z['selected_step']]['boxes']);counts['own_loss_selection']+=1
        assert all(torch.equal(v,x['source_state'][n]) for n,v in ep['initial_state'].items())
        assert torch.count_nonzero(fit['initial_state']['spatial.query_residual'])==0
        assert len(x['anchors'])<=4 and all(a['weight']==1 for a in x['anchors'])
        assert (x['pixel_sha']==x['clean_pixel_sha'])==(x['condition']=='clean')
        assert not x['anchors'] or fit['backwards']==ep['backwards']==10
        if not x['anchors']:assert fit['backwards']==0 and torch.equal(fit['final']['boxes'],x['before']);counts['empty_exact']+=1
        direct,_=reconstruct(x['native']['boxes'],x['anchors'],x['frame_ids'],'absolute');assert torch.equal(direct,x['direct'])
        counts['same_reference_direct']+=1
        if x['input_reuse']:
            old=load(x['input_reuse']['path']);assert sha(x['input_reuse']['path'])==x['input_reuse']['sha256']
            assert old['pixel_sha']==x['pixel_sha'] and old['anchors']==x['anchors']
            assert old['episodic']['losses']==ep['losses'] and old['temporal']['losses']==tt['losses']
            assert x['cost']['DINO']==0;counts['same_input_stage2_reuses']+=1
        arms={};pair=(x['model'],x['key']);observed_union=sorted(common[pair])
        boxes={'Frozen':x['native']['boxes'],'Frozen_formal':x['native']['boxes'],'Episodic':ep['final']['boxes'],
               'C1':fit['final']['boxes'],'Direct':x['direct'],'C1_Before':x['before'],'C1_fixed10':fit['path'][-1]['boxes'],
               'Episodic_native_time':ep['final']['boxes'],'C1_native_time':fit['final']['boxes'],'Direct_native_time':x['direct']}
        for a,b in boxes.items():
            interval=x['native']['indices'] if a=='Frozen' or a.endswith('_native_time') else x['formal_indices']
            mm=score(b,gt,x['frame_ids'],interval,x['observed'],x['anchors']);ind=metric(b,dict(frame_ids=x['frame_ids'],indices=interval,observed=x['observed']),gt)
            for m,v in ind.items():
                assert (v is None)==(mm[m] is None)
                if v is not None:maximum=max(maximum,abs(v-mm[m]));assert abs(v-mm[m])<1e-9
                counts['independent_metric_values']+=1
            gtvalid=np.asarray(gt['valid'],bool);gtvalid[observed_union]=False
            ious=overlap_np(b,gt['boxes']);mm['common_unobserved_sIoU']=float(ious[gtvalid].mean()) if gtvalid.any() else None
            arms[a]={m:mm[m] for m in METRICS}
        quality=collections.Counter();obsrows=[]
        for (_,pos),o in x['expert']['observations'].items():
            probe=o['probe'];valid=bool(gt['valid'][pos]);qs=None
            if valid:
                cs=np.asarray(probe['boxes']).reshape(-1,4);qs=overlap_np(cs,np.tile(np.asarray(gt['boxes'][pos]),(len(cs),1))).tolist()
                quality['valid_observations']+=1
                quality['rejected_best_IoU_ge05']+=not probe['accepted'] and bool(qs) and max(qs)>=.5
                if probe['accepted']:
                    j=int(np.argmax(probe['target_scores']));quality['accepted_IoU_lt03']+=qs[j]<.3
            obsrows.append(dict(position=pos,accepted=probe['accepted'],reason=probe['reason'],GT_valid=valid,candidate_IoUs=qs))
        rows.append(dict(key=x['key'],source=x['source'],model=x['model'],condition=x['condition'],stream=x['stream'],ordinal=x['ordinal'],
            arms=arms,anchors=len(x['anchors']),observations=len(x['observed']),selected_step=fit['selected_step'],
            expert_quality={k:quality[k] for k in ['valid_observations','accepted_IoU_lt03','rejected_best_IoU_ge05']},
            observed=obsrows,cost=x['cost'],file=f,pixel_sha=x['pixel_sha']))
    expected=448 if stage=='homogeneous' else 64;assert len(rows)==expected
    summaries={};contrasts={}
    if stage=='homogeneous':
        for m in MODELS:
            clean={r['key']:r for r in rows if r['model']==m and r['condition']=='clean'}
            for c in CONDITIONS:
                rr=[r for r in rows if r['model']==m and r['condition']==c]
                for scope,ss in [('all32',rr),('after8',[r for r in rr if r['ordinal']>=8])]:
                    summaries[m+'|'+c+'|'+scope]=summarize(ss)
                terms={k:{} for k in ['D_corruption_harm','G_C1_gain','M_memory_gain','G_minus_Gclean','M_minus_Mclean']}
                for metric_ in METRICS:
                    values={k:[] for k in terms}
                    for r in rr:
                        cl=clean[r['key']];aa=r['arms'];bb=cl['arms']
                        d=diff(bb['Frozen'][metric_],aa['Frozen'][metric_]);g=diff(aa['C1'][metric_],aa['Frozen'][metric_]);memory=diff(aa['C1'][metric_],aa['Episodic'][metric_])
                        gc=diff(bb['C1'][metric_],bb['Frozen'][metric_]);mc=diff(bb['C1'][metric_],bb['Episodic'][metric_])
                        for k,v in zip(terms,[d,g,memory,diff(g,gc),diff(memory,mc)]):values[k].append(v)
                    for k,vv in values.items():terms[k][metric_]=aggregate(vv,rr)
                contrasts[m+'|'+c]=terms
    else:
        baseline={(r['model'],r['key']):r for r in read(OUT/'homogeneous/RESULTS.json') if r['condition']=='clean'}
        for m in MODELS:
            rr=[r for r in rows if r['model']==m];summaries[m+'|all32']=summarize(rr)
            for i in range(4):
                seg=[r for r in rr if r['ordinal']//8==i];summaries[m+f'|segment{i}']=summarize(seg)
                if i==3:
                    contrasts[m+'|final_clean_vs_allclean']={a:{mm:aggregate([diff(r['arms'][a][mm],baseline[(m,r['key'])]['arms'][a][mm]) for r in seg],seg) for mm in METRICS} for a in ['C1','C1_Before','Frozen','Episodic']}
                    assert all(r['arms']['Frozen']==baseline[(m,r['key'])]['arms']['Frozen'] for r in seg)
    write(OUT/stage/'RESULTS.json',rows);write(OUT/stage/'SUMMARY.json',summaries);write(OUT/stage/'CONTRASTS.json',contrasts)
    write(OUT/stage/'AUDIT.json',dict(status='passed',counts=dict(counts),maximum_metric_error=maximum,streams=len(previous),created=time.time()))
    print('SCORED',stage,dict(counts),'maxerr',maximum,flush=True)
    if stage=='homogeneous':
        for k,v in contrasts.items():print(k,{term:round(z['vIoU_corrected']['mean']*100,4) for term,z in v.items()},flush=True)
    else:print(contrasts,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['homogeneous','changing']);evaluate(ap.parse_args().phase)
