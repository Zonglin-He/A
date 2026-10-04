"""CPU-only, after-global-seal dense scores and donor-balanced paired summaries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_correction_scope_common_v1 import *
from scripts.tastvg_cpu_handoff_v1 import verify_cpu
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary
from scripts.score_tastvg_negative_evidence_v1 import dense,metadata,add_differences,tails

LIFE_ARMS=('Frozen','A_before','A_after','E_before','E_after')
LIFE_CONTRASTS=[('A_before','Frozen'),('A_after','A_before'),('E_after','E_before'),
                ('E_after','A_after'),('E_after','A_before')]

def old_post(c,old):
    """Read the sealed matched post output where the legacy writer omitted it."""
    import torch
    if 'post_prediction' in old:return old['post_prediction']
    if not c['scheduled']:
        assert not old['updated'] and old['pre_sha']==old['post_sha']
        assert all(torch.equal(v,old['post_state'][n]) for n,v in old['pre_state'].items())
        return old['slow']
    donor=oldchecked(local_payload_path(c));arm=donor['arms']['rank_native']
    assert all(torch.equal(v,donor['pre_state'][n]) for n,v in old['pre_state'].items())
    assert all(torch.equal(v,arm['post_state'][n]) for n,v in old['post_state'].items())
    raw_pre=arm['steps'][0]['pre_prediction'];raw_post=arm['steps'][-1]['post_prediction']
    assert torch.equal(old['slow']['boxes'],raw_pre['boxes'])
    assert torch.equal(torch.as_tensor(old['slow']['indices']),torch.as_tensor(raw_pre['indices']))
    # The top-level negative-evidence readout uses Fast's fixed interval; the
    # per-step output preserves the original free native temporal readout.
    return raw_post

def aggregate(rows,fields,contrasts):
    z=summary(rows,fields)
    z['negative_tails']={a+'_minus_'+b:tails(rows,a,b) for a,b in contrasts}
    z['orders']={o:summary([r for r in rows if r['order']==o],fields) for o in ('order1','order2')}
    return z

def grouped_scope(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        for scope in r['roles']:
            if scope in SCOPES:groups[(r['cell_key'],scope)].append(r)
    out=[]
    fields=['pre_'+m for m in ('v','t','s')]+[b+'_'+m for b in BLOCKS for m in ('v','t','s')]
    fields += [b+'_minus_pre_'+m for b in BLOCKS for m in ('v','t','s')]
    fields += [b+'_gross_'+g+'_pre' for b in BLOCKS for g in ('gain','loss')]
    fields += [b+'_'+setting+'_minus_pre_'+m for b in BLOCKS for setting in ('free','source_fixed') for m in ('v','t','s')]
    for (k,scope),rr in groups.items():
        r=rr[0];meta={f:r[f] for f in ('cell_key','dataset','split','source_id','condition','order','arrival')}
        out.append(dict(**meta,scope=scope,target_count=len(rr),
            **{f:float(np.mean([v[f] for v in rr])) for f in fields}))
    return out

def summarize(life,matrix):
    out=dict(lifecycle={},scope={},cross_corruption={},target_cluster_sensitivity={})
    grouped=grouped_scope(matrix)
    for ds in DATASETS:
        for sp in ('search','confirm'):
            out['lifecycle'].setdefault(ds,{})[sp]={}
            out['scope'].setdefault(ds,{})[sp]={}
            out['cross_corruption'].setdefault(ds,{})[sp]={}
            out['target_cluster_sensitivity'].setdefault(ds,{})[sp]={}
            for group in ('all','clean','corrupt','expert_corrupt','nonexpert_corrupt'):
                rr=[r for r in life if r['dataset']==ds and r['split']==sp and
                    (group=='all' or (r['condition']=='clean')==(group=='clean')) and
                    (group not in ('expert_corrupt','nonexpert_corrupt') or r['expert_scheduled']==(group=='expert_corrupt'))]
                fields=[a+'_'+m for a in LIFE_ARMS for m in ('v','t','s')]
                fields += [a+'_minus_'+b+'_'+m for a,b in LIFE_CONTRASTS for m in ('v','t','s')]
                fields += [a+'_gross_'+g+'_'+b for a,b in LIFE_CONTRASTS for g in ('gain','loss')]
                fields += [a+'_'+setting+'_'+m for a in LIFE_ARMS for setting in ('free','fast') for m in ('v','t','s')]
                out['lifecycle'][ds][sp][group]=aggregate(rr,fields,LIFE_CONTRASTS)
            for condition_group in ('all','clean','corrupt'):
                scopes={}
                for scope in SCOPES:
                    rr=[r for r in grouped if r['dataset']==ds and r['split']==sp and r['scope']==scope and
                        (condition_group=='all' or (r['condition']=='clean')==(condition_group=='clean'))]
                    fields=[b+'_minus_pre_'+m for b in BLOCKS for m in ('v','t','s')]
                    fields += [b+'_gross_'+g+'_pre' for b in BLOCKS for g in ('gain','loss')]
                    fields += [b+'_'+setting+'_minus_pre_'+m for b in BLOCKS for setting in ('free','source_fixed') for m in ('v','t','s')]
                    z=summary(rr,fields);z['orders']={o:summary([r for r in rr if r['order']==o],fields) for o in ('order1','order2')}
                    # Tails are cell-level targets; donor-balanced means remain the primary estimator.
                    raw=[r for r in matrix if r['dataset']==ds and r['split']==sp and scope in r['roles'] and
                        (condition_group=='all' or (r['condition']=='clean')==(condition_group=='clean'))]
                    z['target_cells']=len(raw);z['negative_tails']={b+'_minus_pre':tails(raw,b,'pre') for b in BLOCKS}
                    scopes[scope]=z
                    target_rows=[dict(r,source_id=r['target_source_id']) for r in raw]
                    out['target_cluster_sensitivity'][ds][sp].setdefault(condition_group,{})[scope]=summary(target_rows,fields)
                out['scope'][ds][sp][condition_group]=scopes
                selected=[r for r in matrix if r['dataset']==ds and r['split']==sp and 'different_corruption' in r['roles'] and
                          (condition_group=='all' or (r['condition']=='clean')==(condition_group=='clean'))]
                ff=[b+'_cross_minus_same_v' for b in BLOCKS]
                out['cross_corruption'][ds][sp][condition_group]=summary(selected,ff)
    return out,grouped

def score():
    import torch
    verify_cpu(BASE)
    torch.set_num_threads(2);bar=seal();tick=time.monotonic();start=time.time()
    labels={ds:{sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ('search','confirm')} for ds in DATASETS}
    lock=read(BASE/'RUNTIME_LOCK.json')
    for f,h in lock['GT_inputs'].items():assert sha(ROOT/f)==h
    cohort=read(BASE/'COHORT.json');cells=cohort['cells'];plans={ds:plan(ds) for ds in DATASETS}
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth as vid_truth
    ar=cohort['alternatives'];altrows=[ar[k]['row'] for k in sorted(ar,key=int)];altparents=sorted(map(int,ar))
    alternate_plan=dict(rows=altrows,orders={'one':list(range(len(altrows)))})
    ad,asp,provenance=vid_truth('P1',alternate_plan)
    alternate_labels={parent:dict(truth=ad[j],span=asp[j]) for j,parent in enumerate(altparents)}
    write(BASE/'GT_EXPOSURE.json',dict(time=start,after_global_seal=True,global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        GT_inputs=lock['GT_inputs'],alternative_GT_inputs=provenance,GT_used_for_selection=False,GT_used_for_updates=False))
    life=[];matrix=[]
    for c in cells:
        ds=c['dataset'];row=plans[ds]['rows'][c['parent']];g=labels[ds][c['split']][str(c['parent'])]
        truth={int(k):v for k,v in g['truth'].items()};old=oldcell(c);frozen,_=capture(ds,c['parent'],c['condition'])
        native=frozen['prediction'];fixed=native['physical_interval']
        ap=old_post(c,old);af=dict(ap,indices=old['final_indices'],physical_interval=[row['frame_ids'][old['final_indices'][0]],row['frame_ids'][old['final_indices'][1]]+1])
        ab=dict(old['slow'],indices=old['final_indices'],physical_interval=af['physical_interval'])
        if c['scheduled']:
            z=checked(episode_path(ds,c['parent'],c['condition']))['result'];eb=z['prediction'];ea=z['post_prediction']
            ebf=z['output_prediction'];eaf=z['after_fast_prediction']
        else:eb=ea=ebf=eaf=native
        arms=dict(Frozen=native,A_before=old['slow'],A_after=ap,E_before=eb,E_after=ea)
        fast=dict(Frozen=native,A_before=ab,A_after=af,E_before=ebf,E_after=eaf)
        r=metadata(c)
        for a,p in arms.items():
            r.update({a+'_'+m:v for m,v in dense(p,row,truth,g['span'],ds,fixed).items()})
            r.update({a+'_free_'+m:v for m,v in dense(p,row,truth,g['span'],ds).items()})
            r.update({a+'_fast_'+m:v for m,v in dense(fast[a],row,truth,g['span'],ds).items()})
        add_differences(r,LIFE_CONTRASTS);life.append(r)
        if not c['scheduled']:continue
        x=checked(matrix_path(c));byparent={}
        for j,q in enumerate(x['targets']):
            t=q['target'];alt=t['alternative']
            tr=ar[str(c['parent'])]['row'] if alt else plans[ds]['rows'][t['parent']]
            tg=alternate_labels[c['parent']] if alt else labels[ds][c['split']][str(t['parent'])]
            tt={int(k):v for k,v in tg['truth'].items()};interval=q['before']['physical_interval']
            r=dict(**metadata(c),target_index=j,target_source_id=t['parent'],target_condition=t['condition'],
                target_arrival=t['arrival'],roles=t['roles'],alternative_query=alt,cosine=t.get('cosine'),single_write=True)
            r.update({'pre_'+m:v for m,v in dense(q['before'],tr,tt,tg['span'],ds,interval).items()})
            for b in BLOCKS:
                r.update({b+'_'+m:v for m,v in dense(q['after'][b],tr,tt,tg['span'],ds,interval).items()})
                for setting,ii in [('free',None),('source_fixed',q['source_interval'])]:
                    bm=dense(q['before'],tr,tt,tg['span'],ds,ii)
                    pm=dense(q['after'][b],tr,tt,tg['span'],ds,ii)
                    r.update({b+'_'+setting+'_minus_pre_'+m:pm[m]-bm[m] for m in bm})
            add_differences(r,[(b,'pre') for b in BLOCKS]);matrix.append(r)
            if 'matched_cross_baseline' in r['roles']:byparent['same']=r
            if 'different_corruption' in r['roles']:byparent['cross']=r
        assert byparent['same']['target_source_id']==byparent['cross']['target_source_id']
        for b in BLOCKS:byparent['cross'][b+'_cross_minus_same_v']=byparent['cross'][b+'_minus_pre_v']-byparent['same'][b+'_minus_pre_v']
        if len(life)%24==1:print('CPU_SCORE',len(life),1152,flush=True)
    assert len(life)==1152 and sum(r['expert_scheduled'] for r in life)==288
    summaries,grouped=summarize(life,matrix)
    cfg=dict(bundles=BUNDLES,parameters=1792,rho=.05,student_temperature=1.,direction_count=4,
        arrivals=1152,expert_writes=288,scope_target_cells=len(matrix),donor_balanced_scope_rows=len(grouped),
        availability=cohort['availability'],blocks=BLOCKS,scopes=SCOPES,life_arms=LIFE_ARMS,life_contrasts=LIFE_CONTRASTS,
        new_expert_calls=0,new_alternative_query_inputs=len(ar)*6,full_reset=True,GT_used_for_updates=False,
        GT_used_for_selection=False,historically_exposed=True,production_promoted=False,
        primary_lifecycle_interval='source-native fixed for all arms',primary_scope_interval='target common donor-pre native fixed',
        confirm_A_after_source='sealed rank_native final-step raw post output, excluding its top-level Fast interval override; original pre/post states and pre prediction verified exactly; nonexpert no-op uses slow',
        transfer_is_isolated=True,blocks_nonadditive=True,bootstrap_seed=20261004,bootstrap_draws=10000,
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'))
    resources={ds:{stage:read(BASE/ds/(stage+'_RESOURCES.json')) for stage in ('prepare','episodic','matrix')} for ds in DATASETS}
    resources['score']=dict(CPU_wall_seconds=time.monotonic()-tick,CUDA_initialized=torch.cuda.is_initialized(),new_GPU_calls=0)
    for name,value in [('CONFIG',cfg),('LIFECYCLE_ROWS',life),('SCOPE_ROWS',matrix),('DONOR_SCOPE_ROWS',grouped),('SUMMARY',summaries),('RESOURCES',resources)]:write(PUB/(name+'.json'),value)
    write(PUB/'PREDICTION_BARRIER.json',dict(GT_read=False,time=bar['time'],score_start_time=start,
        global_barrier_sha256=cfg['global_barrier_sha256'],runtime_lock_sha256=cfg['runtime_lock_sha256']))
    write(BASE/'SCORE_COMPLETION.json',dict(status='scored_pending_audit',time=time.time(),scope_target_cells=len(matrix)))
    print('SCORED',len(life),len(matrix),len(grouped),flush=True)

if __name__=='__main__':score()
