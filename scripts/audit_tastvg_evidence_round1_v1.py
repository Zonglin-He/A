"""Additional no-GT CPU audits and disjoint pilot/extension component readback."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.score_tastvg_evidence_vulnerability_v1 import metrics,self_metrics,numeric,strong
BASE=ROOT/'artifacts/tastvg_evidence_vulnerability_v1'

def loader_identity():
    errors=[]
    for cp in sorted((BASE/'no_gt_v2/capture').glob('*.pt')):
        a=load(BASE/'capture'/cp.name);b=load(cp)
        for x,y in zip(a['views'],b['views']):assert torch.equal(x['H'],y['H'])
        for x,y in zip(a['evidence'],b['evidence']):
            for k in x:
                if torch.is_tensor(x[k]):assert torch.equal(x[k],y[k]),(cp.name,k)
                else:assert x[k]==y[k]
        errors.append(dict(key=b['key'],original_sha256=sha(BASE/'capture'/cp.name),clean_sha256=sha(cp)))
    assert len(errors)==16
    write(BASE/'no_gt_v2/LOADER_NUMERIC_IDENTITY.json',dict(status='pass',all_H_and_evidence_bitwise_equal=True,queries=16,records=errors))

def main():
    torch.set_num_threads(4)
    loader_identity()
    out=BASE/'full64_v1';rr=read(out/'round1_readback/ROWS.json');p=read(out/'LOCK.json')
    pilot={r['key'] for r in read(BASE/'no_gt_v2/LOCK.json')['rows']}
    checks=[];gradient_norms=[];reinsert=[];records=[]
    def check(a,b):
        if a is None or b is None:assert a is None and b is None;return
        error=abs(float(a)-float(b));assert error<1e-8,(a,b);checks.append(error)
    for row in p['rows']:
        stem=row['key'].replace(':','_');base=load(out/'capture'/f'{stem}.pt')
        for rho in p['config']['rhos']:
            z=load(out/'attack'/f'{stem}_r{rho:g}.pt');rnd=z['random'];mm=[];losses=[]
            for idx,(b,n) in enumerate(zip(base['evidence'],rnd['evidence'])):
                m,l=metrics(b,n);mm.append(m);losses.append(l)
                for k,v in m.items():check(v,numeric(rnd['metrics'][idx][k]))
            check(np.mean(losses),rnd['surrogate'])
            pm=self_metrics(base['prediction'],rnd['prediction'])
            for k,v in pm.items():check(v,rnd['preservation'][k])
            assert abs(rnd['delta_norm']-z['cap']) <= max(1e-8,1e-6*z['cap'])
            for path in z['path'][:-1]:
                assert math.isfinite(path['gradient_norm']);gradient_norms.append(path['gradient_norm'])
            if z.get('reinsertion'):assert z['reinsertion']['full_pipeline_exact'];reinsert.append((row['key'],rho))
            records.append(dict(key=row['key'],rho=rho,random_strong_preserved=bool(pm['strict'] and pm['self_vIoU']>.95 and strong(mm))))
    assert len(gradient_norms)==1920 and len(reinsert)==6
    subsets={}
    for name,subset in [('pilot16',[r for r in rr if r['key'] in pilot]),('extension48',[r for r in rr if r['key'] not in pilot]),('all64',rr)]:
        stats={}
        for rho in (.005,.01,.02):
            rows=[r for r in subset if r['rho']==rho];n=len(rows)
            types={t:0 for t in ['TTS','ASA','Query','selection']}
            for r in rows:
                for t in types:
                    qualified=any(v is not None and ((t=='TTS' and k.endswith('_JSD') and v>=.01) or (t=='ASA' and k.startswith('ASA') and k.endswith('cosine_drift') and v>=.1) or (t=='Query' and k.startswith('Q') and k.endswith('cosine_drift') and v>=.1) or (t=='selection' and k.endswith('jaccard') and v<=.5)) for m in r['metrics'] for k,v in m.items())
                    types[t]+=int(qualified and r['preservation']['strict'] and r['preservation']['self_vIoU']>.95)
            stats[str(rho)]=dict(n=n,strong=sum(r['strong_preserved'] for r in rows),types=types,self_vIoU_mean=np.mean([r['preservation']['self_vIoU'] for r in rows]),self_vIoU_min=min(r['preservation']['self_vIoU'] for r in rows),random_strong=sum(r['random_strong_preserved'] for r in records if r['rho']==rho and r['key'] in {x['key'] for x in rows}),terminal_strict=sum(r['terminal_preservation']['strict'] for r in rows))
        unique={r['key'] for r in subset if r['strong_preserved']}
        random_unique={r['key'] for r in records if r['random_strong_preserved'] and r['key'] in {x['key'] for x in subset}}
        subsets[name]=dict(unique_strong=len(unique),unique_random_strong=len(random_unique),by_rho=stats)
    write(out/'round1_readback/SUPPLEMENTAL_AUDIT.json',dict(status='pass',random_comparisons=len(checks),random_max_error=max(checks),all_gradient_steps_finite=True,gradient_steps=len(gradient_norms),zero_gradients=sum(g==0 for g in gradient_norms),min_gradient_norm=min(gradient_norms),full_reinsertions=reinsert,GT_read=False,script_sha256=sha(Path(__file__))))
    write(out/'round1_readback/SUBSETS.json',subsets)
    print(json.dumps(subsets,indent=2))
if __name__=='__main__':main()
