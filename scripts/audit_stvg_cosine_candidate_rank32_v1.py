"""Second independent scalar-loop audit of cached E-relative ranking diagnostics."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_cosine_candidate_headroom32_v1 import D,P,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
p=verify();z=read(P/'RESULTS_PUBLIC.json');checks=collections.Counter();mechanism={}
for ds in ['hc2','vidstg']:
    rates=collections.defaultdict(list);counts=collections.Counter()
    for r in [q for q in z['rows']if q['dataset']==ds]:
        o=load(OLD/'predictions'/(r['alias']+'.pt'));d=load(D/'diagnostics'/(r['alias']+'.pt'));raw=o['direct']['feedback']['rewards'].numpy();prob=o['direct']['feedback']['prob'].numpy();vv=o['direct']['feedback']['valid'].numpy().copy();vv[:,0]=True;eligible=[]
        for f in r['frames']:
            t=f['position'];g=d['quality'][t]-d['base_quality'][t];good=[i for i in range(len(g))if vv[t,i]and g[i]>1e-10];bad=[i for i in range(len(g))if vv[t,i]and g[i]<-1e-10];sg=[i for i in range(len(g))if vv[t,i]and g[i]>=.05];sb=[i for i in range(len(g))if vv[t,i]and g[i]<=-.05]
            for prefix,score in [('raw',raw[t]),('posterior',prob[t])]:
                for key,G,B in [(prefix+'_auc',good,bad),('meaningful_'+prefix+'_auc',sg,sb)]:
                    if not G or not B or not r['active_teacher']:assert f[key]is None
                    else:
                        x=sum(1 if score[i]-score[j]>1e-12 else .5 if abs(score[i]-score[j])<=1e-12 else 0 for i in G for j in B)/(len(G)*len(B));assert abs(x-f[key])<1e-12
                    checks['independent_pair_ranking_statistics']+=1
                if r['active_teacher']:
                    above=any(score[i]>score[0]+1e-12 for i in good);assert above==f['any_good_above_E_'+prefix];checks['independent_E_relative_ranking']+=1
            if r['active_teacher']and f['has_gain_5pp']:eligible.append(f)
        if eligible:
            fields=dict(raw_rejects_all_good=lambda f:f['strong_pool_no_good_outranks_E_raw'],raw_good_but_prior_blocks_all=lambda f:not f['strong_pool_no_good_outranks_E_raw']and f['strong_pool_no_good_outranks_E_posterior'],posterior_top_good_update_nonpositive=lambda f:f['posterior_good_but_On_nonpositive'])
            for key,fun in fields.items():
                c=sum(bool(fun(f))for f in eligible);counts[key]+=c;rates[key].append(c/len(eligible))
    mechanism[ds]=dict(query_count=len(next(iter(rates.values()))),counts=dict(counts),query_balanced_rates={k:float(np.mean(v))for k,v in rates.items()},note='conditional on >=5IoU-pp headroom and active teacher; causes not mutually exclusive')
verify();put(P/'ROOT_RANKING_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),script_sha256=sha(Path(__file__)),mechanism=mechanism));print(dict(checks),flush=True)
a=read(D/'PUBLIC_FILES_FINAL.json');a['files']+=['scripts/audit_stvg_cosine_candidate_rank32_v1.py',str((P/'ROOT_RANKING_AUDIT.json').relative_to(ROOT))];put(D/'PUBLIC_FILES_FINAL.json',a,True)
