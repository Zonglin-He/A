"""F39 finite user-defined protocol, never modifies an A4 registry or result."""
import copy
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/decota_dense_support_v1'
F38=ROOT/'artifacts/decota_spatial4_weighted_attribute_v1'
F36=ROOT/'artifacts/decota_simplification_partial_v1'

def plan():
    p=read(OUT/'LOCK.json')
    for f,h in p['protected_pins'].items(): assert sha(ROOT/f)==h,f
    return p

def main():
    old=read(F38/'LOCK.json');seal=read(F38/'COMPLETION.json')
    protected={**old['protected_pins'],**seal['files'],**seal['code_pins']}
    for f in ['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json',
              'artifacts/decota_spatial4_weighted_attribute_v1/COMPLETION.json']:
        protected[f]=sha(ROOT/f)
    for f in (ROOT/'methods/decota_spatial4_v1').glob('*'):
        if f.is_file():protected[str(f.relative_to(ROOT))]=sha(f)
    rows=copy.deepcopy(old['rows'])
    for c, rr in rows.items():
        dev=[r for r in rr if r['f37_role']=='development']
        assert len(dev)==8 and len(rr)==24 and len({r['group'] for r in rr})==24
        for i,r in enumerate(rr):
            donor=next(d for d in dev[i%8:]+dev[:i%8] if d['group']!=r['group'])
            r['F39_wrong_query']=dict(key=donor['key'],source=donor['source'],
                caption=donor['input']['caption'],subject=donor['subject'])
            r['F39_roll']='floor(n_valid/2), separately per original offset, positive direction'
            f=F36/'space'/(r['key'].replace(':','_')+'.pt')
            protected[str(f.relative_to(ROOT))]=sha(f)
    for f,h in protected.items():assert sha(ROOT/f)==h,f
    p=dict(name='DeCoTA_F39_A4_dense_event_support',created_unix=time.time(),rows=rows,protected_pins=protected,
        attachment=dict(path='./private_authorization_notes/authorization.txt',
            sha256=sha('./private_authorization_notes/authorization.txt')),
        spatial=dict(version='exact frozen F38 A4',reuse='F36 actual A4 states and full-grid boxes, SHA checked',
            parameters=1792,steps=10,planned_times=4,views=1,gamma=0,lr={'hcstvg1_test':.01,'vidstg_test':.1},new_DINO=0),
        temporal=dict(parameters=66306,steps=5,lr=[1e-4,1e-3,1e-2],beta=[.1,1.],default=dict(lr=.001,beta=.1),
            teacher='matched native FP32 final pred_actioness logits; exactly one sigmoid; all valid positions',
            loss='physical-cell weighted soft Bernoulli BCE(a,r(P_phi)) + beta KL(P0||P_phi), offset mean',
            cells='F25 exact observed per-offset support [first frame_id,last frame_id+1); physical midpoint cells, normalized; no exterior background',
            epsilon=1e-12,optimizer='AdamW betas(.9,.999),eps1e-4,weight_decay0',
            proposal_alpha=[1.,.5,.25,.125],acceptance='actual objective decreases >1e-9; reject restores optimizer and parameters; save initial and best actual state',
            selection='One common config maximizing equally weighted two-direction source-macro delta_v on 8+8 dev only; tie <=1e-12 native units prefers default, then |log10(lr/.001)|, beta ascending, lr ascending',
            arms=dict(T0='A4+native',T1='real temporal head TTA',T2='same evidence/loss 5-step Adam independent output logits',
                T3='same pixels and geometry, prelocked wrong query frozen actionness; correct-query network student',
                T4='prelocked within-query time roll teacher; correct-query network student'),
            controls='T0-T4 on all dev and eval; T1 dev winner reused, not rerun',
            output='native s<e decoder per offset then physical envelope; full updated endpoints, fixed A4 boxes',
            GT_online=False,total_network_parameters=68098),
        caps=dict(fits=300,backward=1500,new_DINO=0),
        uncertainty=dict(seed=20260914,bootstrap=10000,neutral_pp=.1,untouched=False,historical_exposure=True),
        deployment='Defined candidate contains both parameter branches; neither CURRENT registry is changed. Effect assessed separately from definition.',
        forbidden=['space modification or experiments','extra backbone/expert','GT adaptation','ASA vocabulary','coverage warm start','crop/warp','new schedule','corruption','new full loop'])
    write(OUT/'LOCK.json',p)
    write(OUT/'STATUS.json',dict(stage='implementation_math_checks',finished=False))
    print('F39 locked', {c:len(v) for c,v in rows.items()},len(protected),'protected files')

if __name__=='__main__':main()
