"""Independent CPU readback of every tube, conditional pair and gate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,itertools,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_e_tube_critic64_v1 import D,P,verify,CRITIC


def boot(a):
    a=np.array(a,float);rng=np.random.default_rng(20261018)
    return np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975])


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official
    from vg_tta.stvg_joint_event_critic_v1 import EventCritic,collate,candidate_tensor
    torch.set_num_threads(4);p=verify(True);r=read(P/'RESULTS.json');assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256']
    targets={d:truths(d,dict(parents=[x['query_ordinal']for x in p['records']if x['dataset']==d]))for d in ['hc2','vidstg']}
    plans={d:read(ROOT/f'artifacts/decota_paper_experiments_v1/{d}/PLAN.json')['rows']for d in targets};models={};counts=collections.Counter()
    # The legacy Vid GT loader sets2 CPU threads on import; restore the
    # original measurement's4 before exact frozen critic replay.
    torch.set_num_threads(4)
    for d in ['hc2','vidstg']:
        sd='vidstg'if d=='hc2'else'hc2';m=EventCritic().eval().requires_grad_(False);m.load_state_dict(load(CRITIC/'critics'/(sd+'.pt'))['state']);models[d]=m
    for rec,res in zip(p['records'],r['rows']):
        assert rec['alias']==res['alias'];z=load(D/'predictions'/(rec['alias']+'.pt'));source=load(ROOT/rec['prior_Source']);e=load(ROOT/rec['prior_prediction']);ds=rec['dataset'];q=rec['query_ordinal']
        assert np.array_equal(z['boxes'][0],e['outputs']['Source_no_correction'])and np.array_equal(z['boxes'][2],e['outputs']['E_DINO_support']);counts['old_Source_E_arrays_exact']+=2
        # Independent direct 2-offset physical-envelope enumeration for top3.
        marginal=[]
        for record,logits in zip(source['records'],source['logits']):
            logits=logits.double().numpy()[0];s,t=np.triu_indices(len(logits),1);v=logits[s,0]+logits[t,1];prob=np.exp(v-np.logaddexp.reduce(v));ids=np.array(record['frame_ids']);marginal.append((ids[s],ids[t]+1,prob))
        mass={}
        for a,b,v in zip(*marginal[0]):
            for c,d,w in zip(*marginal[1]):
                key=(int(min(a,c)),int(max(b,d)));mass[key]=mass.get(key,0.)+float(v*w)
        expected=[list(e['interval'])]
        for I in sorted(mass,key=lambda x:(-mass[x],x)):
            if list(I)not in expected:expected.append(list(I))
            if len(expected)==4:break
        assert expected==z['intervals'];counts['top_native_physical_intervals']+=1
        with torch.no_grad():scores=models[ds](*collate([candidate_tensor(f,I,z['fps'])for f in z['features']for I in z['intervals']])).reshape(3,-1).numpy()
        assert np.array_equal(scores,z['scores'].numpy());counts['critic_scores_exact']+=scores.size
        vv=[]
        for i,boxes in enumerate(z['boxes']):
            mm=[]
            for j,I in enumerate(z['intervals']):
                v=official(boxes,plans[ds][q],targets[ds][0][q],targets[ds][1][q],I,ds)
                assert all(abs(v[k]-res['all_metrics'][i][j][k])<2e-10 for k in ['v','t','s']);counts['metrics']+=3;mm.append(v)
            vv.append(mm)
        for name,axis,ep in [('spatial_ranking_v',0,'v'),('temporal_ranking_v',1,'v'),('temporal_ranking_t',1,'t')]:
            pairs=[];GTties=0;ties=0;correct=0;wrong=0
            for fixed in range(scores.shape[1-axis]):
                for a,b in itertools.combinations(range(scores.shape[axis]),2):
                    ia,ib=((a,fixed),(b,fixed))if axis==0 else((fixed,a),(fixed,b));d=vv[ia[0]][ia[1]][ep]-vv[ib[0]][ib[1]][ep]
                    if abs(d)<=1e-8:GTties+=1;continue
                    delta=(float(scores[ia])-float(scores[ib]))*d;pairs.append(1 if delta>0 else 0 if delta<0 else .5);ties+=int(delta==0);correct+=int(delta>0);wrong+=int(delta<0)
            v=res[name];assert len(pairs)==v['strict_pairs']and GTties==v['GT_ties']and correct==v['correct']and wrong==v['wrong']and ties==v['critic_ties']
            assert v['accuracy']is None if not pairs else abs(np.mean(pairs)-v['accuracy'])<1e-12;counts['conditional_query_rankings']+=1
        assert tuple(res['selection'])==np.unravel_index(scores.argmax(),scores.shape);counts['joint_selection']+=1
    for ds,g in r['groups'].items():
        rr=[x for x in r['rows']if x['dataset']==ds];assert len(rr)==32
        for name,x in g['ranking'].items():
            a=[v[name]['accuracy']for v in rr if v[name]['accuracy']is not None];assert len(a)==x['n']
            if a:assert abs(np.mean(a)-x['mean'])<1e-12 and np.max(abs(boot(a)-x['CI95']))<1e-12
            counts['ranking_parent_CI']+=1
        for arm,x in g['deltas_vs_E'].items():
            for ep,val in x.items():
                a=[100*(v['values'][arm][ep]-v['values']['E_MBR'][ep])for v in rr];assert abs(np.mean(a)-val['mean'])<1e-12 and np.max(abs(boot(a)-val['CI95']))<1e-12;counts['paired_parent_CI']+=1
    gate=all(r['groups'][d]['ranking'][n]['CI95']is not None and r['groups'][d]['ranking'][n]['CI95'][0]>.5 for d in ['hc2','vidstg']for n in ['spatial_ranking_v','temporal_ranking_v'])and all(r['groups'][d]['deltas_vs_E']['Joint']['v']['mean']>=0 for d in ['hc2','vidstg'])and r['macro']['CI95_pp'][0]>0
    assert gate==r['decision']['advance_stage2'];assert not torch.cuda.is_initialized();verify(True)
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(counts),gate=gate,no_new_GPU_or_target_selection=True))
    print(dict(status='pass',checks=dict(counts),advance_stage2=gate))


if __name__=='__main__':main()
