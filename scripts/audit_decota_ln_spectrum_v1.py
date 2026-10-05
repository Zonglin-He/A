"""Independent saved-vector/SVD audit and standalone anonymous public audit."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_decota_ln_spectrum_v1 import BASE,PUB,OLD,NAMES,QUERY,KEYS,verify,digest
from scripts.decota_matrix_common_v1 import write,sha
from scripts.decota_public_result_io_v1 import read
from scripts.decota_ln_spectrum_math_v1 import *

def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
        return sum(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):assert len(a)==len(b);return sum(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,float):assert np.isclose(a,b,rtol=1e-9,atol=1e-11),(a,b)
    else:assert a==b,(a,b)
    return 1

def public(folder):
    p=Path(folder);tick=time.time();blocks=read(p/'SPECTRUM_INPUTS.json');segments=read(p/'PREFIX_INPUTS.json');gs=[];hs=[];ps=[];checks=0
    for b in blocks:
        g=np.array(b['gram']);assert g.shape==(len(b['rows']),)*2
        assert np.allclose(g,g.T,rtol=0,atol=1e-10)
        for i,r in enumerate(b['rows']):assert np.isclose(r['proposal_norm']**2,g[i,i],rtol=1e-9,atol=1e-12)
        gs+=geometry(b);hs+=holdout(b);checks+=g.size
    for s in segments:
        g=np.array(s['gram']);assert np.allclose(g,g.T,rtol=0,atol=1e-10);ps+=prefix(s);checks+=g.size
    checks+=compare(hs,read(p/'HELDOUT_ROWS.json'));checks+=compare(ps,read(p/'PREFIX_ROWS.json'))
    checks+=compare(plain(summarize_geometry(gs,hs,ps)),read(p/'GEOMETRY_SUMMARY.json'))
    episodes={b['dataset']:b for b in blocks if b['stream']=='episodic'}
    pairs=read(p/'PAIR_ROWS.json');seen=set()
    for q in pairs:
        key=(q['dataset'],q['split'],q['condition'],q['donor'],q['recipient']);assert key not in seen;seen.add(key)
        assert q['donor']!=q['recipient'] and q['donor_write_scale']==1/16
        g=np.array(episodes[q['dataset']]['gram']);c=cosine(g,q['donor_gram_index'],q['recipient_gram_index'])
        checks+=compare(c,q['cosine']);assert np.isclose(q['utility_v'],q['before_v']-q['frozen_v'],atol=1e-15)
        checks+=1
    for s in read(p/'UTILITY_SUMMARY.json'):
        q=[r for r in pairs if r['dataset']==s['dataset'] and r['split']==s['split'] and ((r['condition']=='clean')==(s['group']=='clean'))]
        checks+=compare(pair_stats(q),s['summary'])
        for z in s['leave_one_donor_out']:
            rr=[r for r in q if r['donor']!=z['removed_donor']];v=pair_stats(rr,draws=0)
            checks+=compare(v.get('metrics',{}),z['metrics']);assert z['remaining_pairs']==len(rr)
    cost=read(p/'ANALYSIS_COST.json');assert cost['new_model_calls']==cost['new_backward_calls']==cost['new_expert_calls']==cost['new_GT_scoring']==0
    out=dict(status='pass',anonymous_Gram_projection_pair_and_bootstrap_checks=checks,correction_vectors_or_weights_required=False,seconds=time.time()-tick)
    print('LN_PUBLIC_AUDIT',checks,round(time.time()-tick,2),flush=True)
    return out

def root():
    import torch
    from scipy.linalg import svd
    verify();tick=time.time();torch.set_num_threads(2);checks=0;bar=read(OLD/'ONLINE_GLOBAL_BARRIER.json')
    segs=read(PUB/'PREFIX_INPUTS.json');blocks=read(PUB/'SPECTRUM_INPUTS.json');lookup={tuple(s[k] for k in KEYS):s for s in segs};xs={}
    for s in segs:
        key=tuple(s[k] for k in KEYS);x=np.load(BASE/'vectors'/('_'.join(key)+'.npz'))['vectors'];xs[key]=x
        assert digest(x)==s['vector_matrix_sha256'];assert np.allclose(x@x.T,s['gram'],atol=1e-10,rtol=1e-12);checks+=x.shape[0]**2
    epmat={b['dataset']:np.load(BASE/'vectors'/f"{b['dataset']}_episodic.npz")['vectors'] for b in blocks if b['stream']=='episodic'}
    epmeta={b['dataset']:{(m['split'],m['condition'],m['source_id']):i for i,m in enumerate(b['rows'])} for b in blocks if b['stream']=='episodic'}
    initial={};committed={};prevh={};first={};single_prefix={};actual_pairs={}
    for done,(rel,h) in enumerate(sorted(bar['files'].items()),1):
        f=OLD/rel;assert sha(f)==h;p=torch.load(f,map_location='cpu',weights_only=False);k=tuple(p[a] for a in KEYS)
        def cat(s):return np.concatenate([s[n].numpy().astype(float) for n in NAMES])
        a=cat(p['initial']);v=(cat(p['fit']['state'])-a) if p['fit'] else np.zeros(1536);c=cat(p['committed']);s=cat(p['source_state']);source=p['parent']
        if p['stream']=='episodic':x=epmat[p['dataset']][epmeta[p['dataset']][(p['split'],p['condition'],source)]];assert np.array_equal(a,s)
        else:
            x=xs[k][p['arrival']]
            assert np.array_equal(a,committed.get(k,s));assert p['previous_payload_sha256']==prevh.get(k)
            if k not in first and not np.array_equal(a,c):first[k]=dict(donor=source,state=c.copy(),vector=v.copy(),arrival=p['arrival']);single_prefix[k]=True
            elif k in first and single_prefix[k]:
                d=first[k];assert np.array_equal(a,d['state'])
                if source!=d['donor']:actual_pairs[(p['dataset'],p['split'],p['condition'],d['donor'],source)]=True
                if not np.array_equal(a,c):single_prefix[k]=False
            committed[k]=c;prevh[k]=h
        assert np.array_equal(v,x);assert np.count_nonzero(p['initial'][QUERY].numpy())==0;checks+=1536+256
        if done%1152==0:print('LN_ROOT_VECTOR_CHECK',done,13824,flush=True)
    pairs=read(PUB/'PAIR_SELECTIONS.json');assert actual_pairs.keys()=={(q['dataset'],q['split'],q['condition'],q['donor'],q['recipient']) for q in pairs}
    # Independent direct rectangular SVD, rather than the production Gram eigh.
    sg=read(PUB/'GEOMETRY_SUMMARY.json');maxerror=0.
    for b in blocks:
        x=np.load(BASE/'vectors'/f"{b['dataset']}_{b['stream']}.npz")['vectors'];assert digest(x)==b['vector_matrix_sha256']
        for sp in ['search','confirm']:
            for gr in ['corruption','clean']:
                ids=[i for i,m in enumerate(b['rows']) if m['split']==sp and ((m['condition']=='clean')==(gr=='clean'))];xx=x[ids];mm=[b['rows'][i] for i in ids];w=norm_weights(mm)
                target=next(s for s in sg['spectra'] if s['dataset']==b['dataset'] and s['stream']==b['stream'] and s['split']==sp and s['group']==gr)
                for mode in ['raw','unit','centered']:
                    z=xx.copy()
                    if mode=='unit':n=np.linalg.norm(z,axis=1);z=np.divide(z,n[:,None],out=np.zeros_like(z),where=n[:,None]>0)
                    if mode=='centered':z-=np.average(z,weights=w,axis=0)
                    ss=svd(z*np.sqrt(w[:,None]),compute_uv=False,lapack_driver='gesvd')**2
                    old=target['spectra'][mode]
                    for r in RANKS:
                        value=ss[:r].sum()/ss.sum() if ss.sum()>0 else None
                        if value is not None:maxerror=max(maxerror,abs(value-old['energy'][str(r)]));assert np.isclose(value,old['energy'][str(r)],rtol=1e-9,atol=1e-10)
                        checks+=1
    # Direct SVD basis verifies every strict-prefix projection at r8. No model.
    by={(q['dataset'],q['stream'],q['split'],q['condition'],q['order'],q['arrival']):q for q in read(PUB/'PREFIX_ROWS.json')}
    for key,x in xs.items():
        hist=[]
        for i,row in enumerate(lookup[key]['rows']):
            if row['commit_norm']<=0:continue
            q=by[key+(i,)]
            if hist:
                _,ss,vh=svd(x[hist],full_matrices=False,lapack_driver='gesvd');keep=ss**2>max(ss[0]**2*1e-12,1e-30);vh=vh[keep]
                value=np.linalg.norm(vh[:8]@x[i])**2/np.linalg.norm(x[i])**2
                maxerror=max(maxerror,abs(value-q['energy']['8']));assert np.isclose(value,q['energy']['8'],rtol=1e-8,atol=1e-9)
                checks+=len(x[i])
            else:assert q['energy']['8'] is None
            hist.append(i)
    checks+=compare(read(PUB/'PAIR_SELECTIONS.json'),[{k:v for k,v in q.items() if k not in ['before_v','frozen_v','utility_v']} for q in read(PUB/'PAIR_ROWS.json')])
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,all_payloads=13824,independent_rectangular_SVD=True,all_prior_projection_r8_direct_SVD=True,
        exact_first_write_pair_coverage=True,new_GT_scoring=False,max_projection_error=maxerror,seconds=time.time()-tick))
    write(PUB/'PUBLIC_AUDIT.json',public(PUB));print('LN_ROOT_AUDIT',checks,maxerror,round(time.time()-tick,2),flush=True)

if __name__=='__main__':root() if len(sys.argv)==1 or sys.argv[1]=='root' else public(sys.argv[1])
