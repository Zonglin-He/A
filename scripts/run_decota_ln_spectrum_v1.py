"""Finite CPU extraction from receipt-verified Top1 trajectories; no inference."""
import sys,os,time,json,hashlib,collections,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.decota_ln_spectrum_math_v1 import *
OLD=ROOT/'artifacts/decota_identity_commitment_v1'
OLDPUB=ROOT/'results/decota_identity_commitment/2026-10-05'
BASE=ROOT/'artifacts/decota_ln_spectrum_v1'
PUB=ROOT/'results/decota_ln_spectrum/2026-10-05'
QUERY='spatial.query_residual'
NAMES=[f'spatial.layers.5.{layer}.{kind}' for layer in ['norm1','norm3','norm4'] for kind in ['weight','bias']]
OWN=['scripts/decota_ln_spectrum_math_v1.py','scripts/run_decota_ln_spectrum_v1.py','scripts/test_decota_ln_spectrum_v1.py','scripts/audit_decota_ln_spectrum_v1.py',
     'scripts/report_decota_ln_spectrum_v1.py','scripts/publish_decota_ln_spectrum_v1.py','protocols/decota_ln_spectrum_v1.md','docs/decota_ln_spectrum_v1/EXECUTION.md']
KEYS=['dataset','stream','split','condition','order']

def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md'
    line=('**2026-10-05｜Top1 DeCoTA LN correction spectrum：'+event+'。** 单frozen DINO/Native WHEN/四原观察/admitted Top1/Adam.03/joint1792/10步own-loss/当前完整纠正与query重置保持；仅CPU读13824旧封存LN状态，1536选态proposal减prestate与实际1/16写分开，各32开发+16历史曝光确认/一query/双序六条件/12流。原始/单位范数/中心化谱、source互斥投影、严格prior-only前缀和首非零写至第二写之前的1/16单donor效用；全局累计Before−Frozen不当isolated。无新GPU/模型/专家/重放/梯度/GT重评分；GT-derived匿名metric只在几何与pair选择seal后join，非fresh。低秩不证明可迁移，不自动执行GT选rank/shared memory/temporal或恢复旧队列，CURRENT不变。见[协议](</home/wwww/visual grounding/protocols/decota_ln_spectrum_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_ln_spectrum_v1/STATUS.json>)。')
    t=f.read_text();f.write_text(t.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+line+'\n',1)+'\n\n### LN spectrum CPU update\n\n'+line+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as o:
        for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=o,stderr=subprocess.STDOUT,check=True)

def verify():
    lock=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**lock['pins'],**lock['inputs'],**lock['protected']}.items():assert sha(ROOT/p)==h,p
    return lock

def prepare():
    assert read(OLD/'FINAL_COMPLETION.json')['commit']=='38810f55a09d2aec1a384a8b585981ac4eec0781'
    old=read(OLD/'RUNTIME_LOCK.json');inputs=['FINAL_COMPLETION.json','ONLINE_LOCK.json','ONLINE_GLOBAL_BARRIER.json','RUNTIME_LOCK.json','vidstg/PLAN.json','hc2/PLAN.json']
    extra=[OLDPUB/'online/ROWS.json',OLDPUB/'online/ROOT_AUDIT.json',OLDPUB/'online/MATCHED_PERSISTENCE_SUMMARY.json']
    lock=dict(version='decota_ln_spectrum_v1',time=time.time(),pins={p:sha(ROOT/p) for p in OWN},protected=old['protected'],
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [*[OLD/p for p in inputs],*extra]},LN_names=NAMES,LN_dimension=1536,
        ranks=RANKS,donor_write_scale=1/16,source_node_bootstrap=10000,seed=SEED,new_inference=0,new_expert=0,new_training=0)
    write(BASE/'RUNTIME_LOCK.json',lock)
    write(PUB/'CONFIGURATION.json',dict(version=lock['version'],predecessor_commit=read(OLD/'FINAL_COMPLETION.json')['commit'],
        cohort=dict(datasets=['vidstg','hc2'],development_sources_each=32,confirmation_sources_each=16,queries_per_source=1,orders=2,historically_exposed=True,
            conditions=['clean','frame_drop_5','frame_freeze_5','motion_blur_5','occlusion_5','exposure_5'],old_logical_arrivals=13824),
        backbone='TA-STVG official same-domain checkpoint',checkpoint_file_sha256={'vidstg':'5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83','hc2':'47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036'},
        spatial_expert='one frozen Grounding DINO',temporal='frozen Native interval',critic='admitted-frame Top1 energy',optimizer=dict(name='Adam',lr=.03,steps=10,selection='first own-loss minimum'),
        current_parameters=1792,LN_dimension=1536,LN_names=NAMES,write_scale=1/16,ranks=RANKS,seed=SEED,new_model_calls=0,new_expert_calls=0,new_GT_scoring=0,
        utility_scope='exact first nonzero-write prefixes at scale1/16 only; source-initialized donor and recipient correction cosine',production_promoted=False))
    status(BASE/'STATUS.json',dict(status='locked_pending_CPU_extraction',time=time.time(),new_inference=0))
    archive('配置/代码/input hashes已锁，待无标签CPU提取；不将低秩当memory收益')

def np_state(s):
    assert set(s)==set(NAMES+[QUERY]);assert all(tuple(s[k].shape)==(256,) for k in s)
    return np.concatenate([s[n].numpy() for n in NAMES]).astype(np.float32)

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def meta(p):return {k:p[k] for k in KEYS}|dict(arrival=p['arrival'],source_id=p['parent'],expert=p['expert'],pixel_sha256=p['pixel_sha256'])

def extract():
    import torch
    tick=time.time();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    def forbidden(*a,**kw):raise RuntimeError('No model execution or backward authorized in this CPU audit')
    torch.nn.Module.__call__=forbidden;torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden;torch.cuda.init=forbidden
    verify();bar=read(OLD/'ONLINE_GLOBAL_BARRIER.json');assert bar['arrivals']==13824 and not bar['GT_read']
    chunks={};ep={};sources={};checks=0;allmeta=[];pairs={};first=None;current=None;prev=None;prevsha=None;between=False
    for done,(rel,h) in enumerate(sorted(bar['files'].items()),1):
        f=OLD/rel;assert sha(f)==h,rel;side=read(f.with_suffix('.json'));assert side['sha256']==h and not side['GT_read'] and side['runtime_lock_sha256']==sha(OLD/'RUNTIME_LOCK.json')
        p=torch.load(f,map_location='cpu',weights_only=False);assert not p['GT_read'] and p['query_reset'] and p['Adam_reset']
        m=meta(p);m['payload_sha256']=h;group=tuple(m[k] for k in KEYS);qzero=np.array_equal(p['initial'][QUERY].numpy(),np.zeros(256,np.float32));assert qzero
        s=np_state(p['source_state']);a=np_state(p['initial']);c=np_state(p['committed']);assert np.count_nonzero(p['committed'][QUERY].numpy())==0
        if p['dataset'] in sources:assert np.array_equal(s,sources[p['dataset']])
        else:sources[p['dataset']]=s.copy()
        z=p['fit'];selected=a if z is None else np_state(z['state']);v=selected.astype(float)-a.astype(float)
        if z is not None:
            assert np.array_equal(np_state(z['initial']),a) and not z['GT_used'] and z['arm']=='top1'
            step=z['selected_step'];assert step==min(z['path'],key=lambda q:q['loss'])['step']
            saved=next(q for q in z['path'] if q['step']==step);assert np.array_equal(np_state(saved['state']),selected)
        expect=a+(selected-a)*np.float32(1/16)
        if p['stream']=='episodic':
            assert np.array_equal(a,s) and np.array_equal(p['before'].numpy(),p['native']['boxes'].numpy())
        else:assert np.array_equal(c,expect),'writeback arithmetic'
        if group!=current:
            current=group;prev=None;prevsha=None;first=None;between=False;assert m['arrival']==0
        if p['stream']!='episodic':
            assert np.array_equal(a,s if prev is None else prev)
            assert p['previous_payload_sha256']==prevsha
        m.update(selected_step=0 if z is None else z['selected_step'],fit_empty=bool(z is not None and z['empty']),
            proposal_norm=float(np.linalg.norm(v)),commit_norm=float(np.linalg.norm(c.astype(float)-a.astype(float))),
            selected_delta_sha256=digest(v),pre_LN_sha256=digest(a),commit_LN_sha256=digest(c),before_boxes_sha256=digest(p['before'].numpy()),
            native_interval=p['interval'],old_source_LN_sha256=digest(s))
        if group not in chunks:chunks[group]=dict(rows=[],vectors=[])
        chunk=chunks[group];assert m['arrival']==len(chunk['rows']);chunk['rows'].append(m);chunk['vectors'].append(v);allmeta.append(m)
        ekey=(p['dataset'],p['split'],p['condition'],p['parent'])
        if p['stream']=='episodic':
            if ekey in ep:assert np.array_equal(v,ep[ekey]['vector']) and m['pixel_sha256']==ep[ekey]['meta']['pixel_sha256']
            else:ep[ekey]=dict(vector=v.copy(),meta=m)
        else:
            changed=not np.array_equal(c,a)
            if first is not None and between:
                assert np.array_equal(a,first['commit'])
                if m['source_id']!=first['meta']['source_id']:
                    # Geometry only: metrics are not opened in this stage.
                    pk=(m['dataset'],m['split'],m['condition'],first['meta']['source_id'],m['source_id'])
                    member={k:m[k] for k in KEYS}|dict(arrival=m['arrival'],payload_sha256=h,expert=m['expert'])
                    if pk in pairs:
                        assert pairs[pk]['recipient_before_boxes_sha256']==m['before_boxes_sha256']
                        pairs[pk]['memberships'].append(member)
                    else:pairs[pk]=dict(dataset=m['dataset'],split=m['split'],condition=m['condition'],donor=first['meta']['source_id'],recipient=m['source_id'],
                        donor_proposal_sha256=first['meta']['selected_delta_sha256'],donor_payload_sha256=first['meta']['payload_sha256'],donor_commit_LN_sha256=first['meta']['commit_LN_sha256'],
                        recipient_before_boxes_sha256=m['before_boxes_sha256'],donor_write_scale=1/16,memberships=[member])
                if changed:between=False
            elif first is None and changed:
                assert np.array_equal(a,s);first=dict(meta=m,commit=c.copy(),vector=v.copy());between=True
                donor_ep=ep[ekey];assert np.array_equal(v,donor_ep['vector']),'first write must equal source-initialized episode'
            prev=c.copy();prevsha=h
        checks+=1536*4+256+len(NAMES)*256*(2 if z is not None else 1)
        if done%576==0:
            status(BASE/'STATUS.json',dict(status='extracting_CPU',done=done,total=13824,pid=os.getpid(),seconds=time.time()-tick,new_inference=0))
            print('LN_EXTRACT',done,13824,round(time.time()-tick,2),flush=True)
    assert len(allmeta)==13824 and len(ep)==576
    publicblocks=[];private=BASE/'vectors';private.mkdir(parents=True,exist_ok=True)
    epindex={}
    for ds in ['vidstg','hc2']:
        for stream in ['episodic','online100']:
            selected=[m for m in allmeta if m['dataset']==ds and m['stream']==stream and (stream!='episodic' or m['order']=='order1')]
            vectors=[]
            for m in selected:
                grp=tuple(m[k] for k in KEYS);vectors.append(chunks[grp]['vectors'][m['arrival']])
            x=np.array(vectors);g=x@x.T
            for i,m in enumerate(selected):m['gram_index']=i
            name=f'{ds}:{stream}';publicblocks.append(dict(dataset=ds,stream=stream,rows=selected,gram=g.tolist(),vector_matrix_sha256=digest(x)))
            np.savez_compressed(private/f'{ds}_{stream}.npz',vectors=x)
            if stream=='episodic':
                epindex[ds]={(m['split'],m['condition'],m['source_id']):m['gram_index'] for m in selected}
    segments=[]
    for group,ch in chunks.items():
        if group[1]=='episodic':continue
        x=np.array(ch['vectors']);g=x@x.T
        segments.append(dict(**dict(zip(KEYS,group)),rows=ch['rows'],gram=g.tolist(),vector_matrix_sha256=digest(x)))
        np.savez_compressed(private/('_'.join(group)+'.npz'),vectors=x)
    blocks={b['dataset']:b for b in publicblocks if b['stream']=='episodic'}
    for pk,q in pairs.items():
        ix=epindex[q['dataset']];i=ix[(q['split'],q['condition'],q['donor'])];j=ix[(q['split'],q['condition'],q['recipient'])];b=blocks[q['dataset']];g=np.array(b['gram'])
        assert b['rows'][i]['selected_delta_sha256']==q['donor_proposal_sha256']
        q.update(donor_gram_index=i,recipient_gram_index=j,cosine=cosine(g,i,j),recipient_proposal_norm=b['rows'][j]['proposal_norm'],
            donor_proposal_norm=b['rows'][i]['proposal_norm'],recipient_episodic_payload_sha256=b['rows'][j]['payload_sha256'])
    write(PUB/'SPECTRUM_INPUTS.json',plain(publicblocks));write(PUB/'PREFIX_INPUTS.json',plain(segments));write(PUB/'PAIR_SELECTIONS.json',plain(list(pairs.values())))
    write(PUB/'EXTRACTION_AUDIT.json',dict(status='pass',payloads=13824,source_initialized_unique_inputs=576,LN_scalar_and_chain_checks=checks,
        private_vector_files=len(list(private.glob('*.npz'))),pairs=len(pairs),logical_pair_memberships=sum(len(q['memberships']) for q in pairs.values()),
        model_execution_forbidden=True,new_model_calls=0,new_backward_calls=0,new_expert_calls=0,GT_metric_rows_opened=False,seconds=time.time()-tick))
    files={str(f.relative_to(ROOT)):sha(f) for f in PUB.glob('*.json')}
    write(BASE/'GEOMETRY_INPUT_BARRIER.json',dict(status='sealed',files=files,pairs=len(pairs),old_GT_exposed=True,GT_metrics_joined=False,time=time.time()))
    assert not torch.cuda.is_initialized();status(BASE/'STATUS.json',dict(status='geometry_sealed_pending_analysis',payloads=13824,pairs=len(pairs),time=time.time()))
    print('LN_GEOMETRY_SEALED',len(pairs),round(time.time()-tick,2),flush=True)

def analyze():
    from scripts.decota_public_result_io_v1 import read as public_read
    verify();bar=read(BASE/'GEOMETRY_INPUT_BARRIER.json')
    for p,h in bar['files'].items():assert sha(ROOT/p)==h
    tick=time.time();gs=[];hs=[];ps=[]
    for b in read(PUB/'SPECTRUM_INPUTS.json'):
        gs+=geometry(b);hs+=holdout(b);print('LN_GEOMETRY_ANALYSIS',b['dataset'],b['stream'],flush=True)
    for s in read(PUB/'PREFIX_INPUTS.json'):ps+=prefix(s)
    write(PUB/'HELDOUT_ROWS.json',plain(hs));write(PUB/'PREFIX_ROWS.json',plain(ps));write(PUB/'GEOMETRY_SUMMARY.json',plain(summarize_geometry(gs,hs,ps)))
    write(BASE/'GEOMETRY_ANALYSIS_BARRIER.json',dict(status='sealed',files={str(f.relative_to(ROOT)):sha(f) for f in [PUB/'HELDOUT_ROWS.json',PUB/'PREFIX_ROWS.json',PUB/'GEOMETRY_SUMMARY.json']},time=time.time(),GT_metrics_joined=False))
    # No labels, annotations or model payload predictions are rescored: only
    # join already-published, independently audited scalar GT-derived metrics.
    exposure=dict(time=time.time(),geometry_input_barrier_sha256=sha(BASE/'GEOMETRY_INPUT_BARRIER.json'),geometry_analysis_barrier_sha256=sha(BASE/'GEOMETRY_ANALYSIS_BARRIER.json'),historically_exposed=True,new_annotation_access=False,new_GT_scoring=False)
    write(BASE/'METRIC_JOIN.json',exposure)
    rows=public_read(OLDPUB/'online/ROWS.json');index={tuple(r[k] for k in KEYS)+(r['arrival'],):r for r in rows};episode={(r['dataset'],r['split'],r['condition'],r['source_id']):r for r in rows if r['stream']=='episodic' and r['order']=='order1'}
    pp=[]
    for q in read(PUB/'PAIR_SELECTIONS.json'):
        vals=[]
        for m in q['memberships']:
            r=index[tuple(m[k] for k in KEYS)+(m['arrival'],)];assert r['payload_sha256']==m['payload_sha256'] and r['source_id']==q['recipient']
            e=episode[(q['dataset'],q['split'],q['condition'],q['recipient'])];assert e['payload_sha256']==q['recipient_episodic_payload_sha256']
            assert e['before_v']==e['frozen_v'] and r['frozen_v']==e['frozen_v']
            vals.append(dict(before_v=r['before_v'],frozen_v=r['frozen_v'],utility_v=r['before_v']-r['frozen_v']))
        assert all(v==vals[0] for v in vals),'deduplicated pair metrics differ'
        pp.append({**q,**vals[0]})
    write(PUB/'PAIR_ROWS.json',plain(pp));summ=[]
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            for group in ['corruption','clean']:
                q=[p for p in pp if p['dataset']==ds and p['split']==sp and ((p['condition']=='clean')==(group=='clean'))]
                stats=pair_stats(q);loo=[]
                for d in sorted({p['donor'] for p in q}):
                    qq=[p for p in q if p['donor']!=d];z=pair_stats(qq,draws=0);loo.append(dict(removed_donor=d,remaining_pairs=len(qq),metrics=z.get('metrics',{})))
                summ.append(dict(dataset=ds,split=sp,group=group,summary=stats,leave_one_donor_out=loo))
    write(PUB/'UTILITY_SUMMARY.json',plain(summ))
    write(PUB/'OLD_MATCHED_PERSISTENCE.json',public_read(OLDPUB/'online/MATCHED_PERSISTENCE_SUMMARY.json'))
    write(PUB/'ANALYSIS_COST.json',dict(cpu_wall_seconds=time.time()-tick,new_model_calls=0,new_GT_scoring=0,new_backward_calls=0,new_expert_calls=0,
        model_state_updated=False,projected_memory_executed=False,utility_is_full_delta_intervention=False))
    status(BASE/'STATUS.json',dict(status='analyzed_pending_independent_audit_report_publication',time=time.time(),pairs=len(pp),new_inference=0))
    archive('13824旧状态提取与label-blind几何封存后匿名效用join已实际完成；待独立算术/公开审计和图报告')
    print('LN_ANALYSIS_COMPLETED',len(pp),round(time.time()-tick,2),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        import traceback
        BASE.mkdir(parents=True,exist_ok=True)
        f=BASE/f'FAILURE_{sys.argv[1]}_{int(time.time())}.json';write(f,dict(stage=sys.argv[1],traceback=traceback.format_exc(),time=time.time(),scientific_outputs_preserved=True))
        raise
