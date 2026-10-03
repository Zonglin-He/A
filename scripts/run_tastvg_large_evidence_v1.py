"""Fixed-support CPU signal seal followed by cached labelled diagnosis."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,collections,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_large_evidence_math_v1 import *
BASE=ROOT/'artifacts/tastvg_large_correction_evidence_v1'
PUBLIC=ROOT/'results/tastvg_large_correction_evidence/2026-10-03'
PRIOR=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
LABELS=ROOT/'results/tastvg_temporal_boundary_support/2026-10-03'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEWS=ROOT/'artifacts/tastvg_current_correction_views_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def write(p,x,mutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert mutable or not p.exists(),p
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def status(s,**kw):write(BASE/'STATUS.json',dict(status=s,time=time.time(),**kw),True)
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def paths(c):
    ds=c['dataset'];cond=c['condition'];parent=c['parent'];p=prefix(c)
    return dict(pred=PRIOR/ds/'predictions'/f'{p}.json',
        head_receipt=POOL/ds/'capture'/cond/f'{parent:05}.json',
        expert_receipt=POOL/ds/'experts/temporal'/cond/f'{parent:05}.json',
        view_receipt=VIEWS/ds/'views'/f'{p}.json')
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
    p=str(args[0])
    if any(k in p for k in ['GT_LABELS','GT_EXPOSURE','test_annotations','valv2_proc',
          '/ROWS.json','/SUMMARY.json','INTERVENTION_ROWS','ORACLE_ROWS']):
        raise PermissionError('Signal worker cannot open GT or labelled metrics')
    if '/results/' in p and not p.startswith(str(PUBLIC)):
        raise PermissionError('Signal worker cannot read previous public results')

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists();sys.addaudithook(guard)
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='completed_and_verified_publication'
    assert read(ROOT/'artifacts/tastvg_temporal_local_oracle_v1/FINAL_COMPLETION.json')['commit']=='ac2102e3527e8344241a62db1fb4c3eceec48991'
    cells=read(PRIOR/'COHORT.json')['cells'];assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    inputs={};counts=collections.Counter()
    def bind(f,expected=None):
        s=str(f.relative_to(ROOT))
        if s not in inputs:inputs[s]=sha(f)
        if expected:assert inputs[s]==expected,f
    for f in [ROOT/'methods/CURRENT_METHOD.json',PRIOR/'COHORT.json',PRIOR/'GLOBAL_PREDICTION_BARRIER.json']:
        bind(f)
    oldpins=read(PRIOR/'RUNTIME_LOCK.json')['inputs']
    for ds in ['vidstg','hc2']:bind(VIEWS/ds/'PLAN.json')
    for c in cells:
        pp=paths(c);bind(pp['pred']);counts[c['dataset']+'/'+c['split']+'/arrivals']+=1
        if not c['scheduled']:continue
        z=read(pp['pred']);assert z['GT_read'] is False and z['cell_key']==key(c)
        assert z['candidates'][z['choices']['A8']]['indices']==z['A_indices']
        for name in ['head_receipt','expert_receipt','view_receipt']:bind(pp[name])
        h=read(pp['head_receipt']);e=read(pp['expert_receipt']);v=read(pp['view_receipt'])
        assert h['pixel_sha256']==e['pixel_sha256']==v['pixel_sha256']==c['pixel_sha256']
        hf=POOL/c['dataset']/h['cache'];ef=POOL/c['dataset']/'experts'/e['cache'];vf=ROOT/v['cache']
        bind(hf,h['sha256']);bind(ef,e['cache_sha256']);bind(vf,v['cache_sha256'])
        for f in [pp['head_receipt'],pp['expert_receipt'],hf,ef]:
            assert oldpins[str(f.relative_to(ROOT))]==inputs[str(f.relative_to(ROOT))],f
        assert v['phase']==.25 and v['GT_read'] is False
        counts[c['dataset']+'/'+c['split']+'/experts']+=1
    pins=['protocols/tastvg_large_correction_evidence_v1.md',
        'docs/tastvg_large_correction_evidence_v1/EXECUTION.md',
        'scripts/tastvg_large_evidence_math_v1.py','scripts/run_tastvg_large_evidence_v1.py',
        'scripts/audit_tastvg_large_evidence_public_v1.py','scripts/report_tastvg_large_evidence_v1.py',
        'scripts/test_tastvg_large_evidence_v1.py']
    write(BASE/'PRIVATE_INPUT_BINDINGS.json',inputs)
    config=dict(version='tastvg_large_correction_evidence_v1',predecessor_commit='ac2102e3527e8344241a62db1fb4c3eceec48991',
        supports=[8,32],signals=SIGNALS,bandwidth_seconds=.5,large_radius=.5,dominant_balance=.25,
        decision='unique_strict_improvement_over_same_A8_else_anchor',tie_epsilon=EPS,
        native_logits_scope='frozen_source_checkpoint_not_current_A',
        native_mapping='equal_offset_mass_interleaved_marginals_not_native_envelope_likelihood',
        UVTG_scope='equal_proposal_boundary_mass_without_confidence',
        two_view_scope='same_model_phases_0_and_025_not_independent_experts',
        groups=GROUPS,new_model_calls=0,new_expert_calls=0,new_GPU_jobs=0,
        GT_use='join_already_published_dense_candidate_metrics_after_new_signal_seal',
        fresh_test=False,confirmation_selects_rules=False,parameters_unchanged=True,
        source_bootstrap_draws=10000,seed=20261003,
        original_parameters={ds:read(VIEWS/ds/'PLAN.json')['params'] for ds in ['vidstg','hc2']},
        checkpoint_state_sha256={ds:read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'] for ds in ['vidstg','hc2']},
        production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        private_input_files=len(inputs),time=time.time())
    write(PUBLIC/'CONFIG.json',config);write(BASE/'COHORT.json',dict(cells=cells,counts=dict(counts)))
    write(BASE/'RUNTIME_LOCK.json',dict(config_sha256=sha(PUBLIC/'CONFIG.json'),
        input_bindings_sha256=sha(BASE/'PRIVATE_INPUT_BINDINGS.json'),
        cohort_sha256=sha(BASE/'COHORT.json'),pins={p:sha(ROOT/p) for p in pins},time=time.time()))
    status('prepared_pending_label_free_signals',GT_read=False,counts=dict(counts))
    print('LARGE_EVIDENCE_LOCKED',json.dumps(dict(counts)),flush=True)

def verify(full=False):
    k=read(BASE/'RUNTIME_LOCK.json')
    for f in sorted((BASE/'revisions').glob('*.json')):
        r=read(f);assert r['science_changed'] is False;k['pins'].update(r['pin_overrides'])
    for p,h in k['pins'].items():assert sha(ROOT/p)==h,p
    assert sha(PUBLIC/'CONFIG.json')==k['config_sha256'] and sha(BASE/'COHORT.json')==k['cohort_sha256']
    assert sha(BASE/'PRIVATE_INPUT_BINDINGS.json')==k['input_bindings_sha256']
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==read(PUBLIC/'CONFIG.json')['production_method_sha256']
    if full:
        for p,h in read(BASE/'PRIVATE_INPUT_BINDINGS.json').items():assert sha(ROOT/p)==h,p
    return k

def signals():
    import torch,functools
    sys.addaudithook(guard);torch.set_num_threads(2);verify();tick=time.time()
    status('label_free_signals_running',GT_read=False,worker_pid=os.getpid())
    bindings=read(BASE/'PRIVATE_INPUT_BINDINGS.json');seen=set()
    def load(p):
        name=str(p.relative_to(ROOT))
        if name not in seen:assert sha(p)==bindings[name];seen.add(name)
        return torch.load(p,map_location='cpu',weights_only=False)
    @functools.lru_cache(maxsize=1)
    def head(path):
        d=load(Path(path));ids=d['frame_ids'];pos=[[ids.index(f) for f in r['frame_ids']] for r in d['records']]
        return merged_logprior([x.numpy() for x in d['prediction']['logits']],pos,len(ids)),ids,d['prediction']['indices'],d['checkpoint_state_sha256'],pos
    rows=[];checks=collections.Counter()
    for c in read(BASE/'COHORT.json')['cells']:
        if not c['scheduled']:continue
        pp=paths(c);z=read(pp['pred']);h=read(pp['head_receipt']);er=read(pp['expert_receipt']);vr=read(pp['view_receipt'])
        lp,ids,nidx,checkpoint,offsets=head(str(POOL/c['dataset']/h['cache']))
        assert checkpoint==read(PUBLIC/'CONFIG.json')['checkpoint_state_sha256'][c['dataset']]
        e0=load(POOL/c['dataset']/'experts'/er['cache']);e1=load(ROOT/vr['cache'])
        lo,hi=ids[0],ids[-1]+1;span=hi-lo;a=z['choices']['A8']
        xy=[[float((p['physical_interval'][0]-lo)/span),float((p['physical_interval'][1]-lo)/span)] for p in z['candidates']]
        assert xy==z['intervals_normalized'] and len(xy)==32
        assert [[ids[i],ids[j]+1] for i,j in [p['indices'] for p in z['candidates']]]==[p['physical_interval'] for p in z['candidates']]
        assert abs(e0['duration']-z['duration_seconds'])<1e-9
        mapped=np.asarray(e1['seconds_segments'])*e1['fps']+e1['physical_origin']
        np.testing.assert_allclose(mapped,e1['proposals'],rtol=1e-6,atol=1e-5)
        ev=dict(cell_key=key(c),dataset=c['dataset'],split=c['split'],source_id=c['parent'],
            condition=c['condition'],order=c['order'],arrival=c['arrival'],anchor_index=a,
            A_state_pre_sha256=z['persistent_pre_sha'],A_state_post_sha256=z['persistent_post_sha'],
            intervals=xy,candidate_indices=[p['indices'] for p in z['candidates']],
            duration_seconds=z['duration_seconds'],bandwidth_normalized=.5/z['duration_seconds'],
            native_logprior=lp.tolist(),native_offset_positions=offsets,
            native_source_indices=nidx,native_A_indices=z['native_indices'],
            source_native_agrees_A_native=nidx==z['native_indices'],source_native_agrees_A8=nidx==z['A_indices'],
            proposals_view0=((np.asarray(e0['proposals']).reshape(-1,2)-lo)/span).tolist(),
            proposals_view1=((np.asarray(e1['proposals']).reshape(-1,2)-lo)/span).tolist(),
            actual_view_observation_difference=vr['actual_observation_difference'],
            bindings=dict(prediction_sha256=sha(pp['pred']),head_sha256=h['sha256'],
                view0_sha256=er['cache_sha256'],view1_sha256=vr['cache_sha256'],pixel_sha256=c['pixel_sha256']),GT_read=False)
        sc,details=scores(ev);ev.update(scores=sc,details=details,
            choices={f'{s}{n}':choose(sc[s],a,n) for s in SIGNALS for n in [8,32]},
            geometry=[geometry(xy[a],v) for v in xy])
        rows.append(ev);checks['expert_cells']+=1;checks['new_scores']+=96
        checks['source_native_agrees_A_native']+=ev['source_native_agrees_A_native'];checks['source_native_agrees_A8']+=ev['source_native_agrees_A8']
        checks['real_two_view_changed']+=vr['actual_observation_difference']>0
        if len(rows)%48==0:
            status('label_free_signals_running',GT_read=False,expert_cells=len(rows),worker_pid=os.getpid())
            print('BOUNDARY_SIGNALS_NO_GT',len(rows),288,flush=True)
    assert len(rows)==288 and not torch.cuda.is_initialized();verify()
    write(PUBLIC/'EVIDENCE_ROWS.json',rows)
    write(PUBLIC/'SIGNAL_SEAL.json',dict(status='sealed',GT_read=False,expert_cells=288,
        file_sha256=sha(PUBLIC/'EVIDENCE_ROWS.json'),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        checks=dict(checks),CUDA_initialized=False,time=time.time()))
    write(BASE/'SIGNAL_RESOURCES.json',dict(CPU_wall_seconds=time.time()-tick,CUDA_initialized=False,
        new_model_calls=0,new_expert_calls=0,weights_opened=0,media_opened=0,time=time.time()))
    status('sealed_pending_cached_GT_metric_join',GT_read=False,expert_cells=288)
    print('SIGNAL_SEAL_COMPLETE',json.dumps(dict(checks)),flush=True)

def diagnose():
    tick=time.time();verify();seal=read(PUBLIC/'SIGNAL_SEAL.json')
    assert seal['status']=='sealed' and seal['GT_read'] is False and seal['file_sha256']==sha(PUBLIC/'EVIDENCE_ROWS.json')
    evidence={e['cell_key']:e for e in read(PUBLIC/'EVIDENCE_ROWS.json')}
    status('cached_GT_metric_join_running');totals=collections.Counter();label_files={}
    for sp in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            f=LABELS/sp/ds/'ROWS.json';label_files[str(f.relative_to(ROOT))]=sha(f);raw=read(f);rows=[];binary=[]
            for r in raw:
                base={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','expert_scheduled','A_state_pre_sha256','A_state_post_sha256']}
                base.update(A8_v=r['A8_v'],A8_t=r['A8_t']);totals['arrivals']+=1
                if r['expert_scheduled']:
                    e=evidence[key(r)];assert e['source_id']==r['source_id']
                    assert e['A_state_pre_sha256']==r['A_state_pre_sha256'] and e['A_state_post_sha256']==r['A_state_post_sha256']
                    assert e['candidate_indices']==r['candidate_indices'] and e['intervals']==r['intervals_normalized']
                    v=np.asarray(r['candidate_v']);t=np.asarray(r['candidate_t']);a=e['anchor_index'];assert abs(v[a]-r['A8_v'])<1e-12
                    base.update(anchor_index=a,candidate_v=v.tolist(),candidate_t=t.tolist(),evidence_key=e['cell_key'])
                    totals['experts']+=1;totals['clean_experts' if r['condition']=='clean' else 'corrupt_experts']+=1
                    for s in SIGNALS:
                        ev=np.asarray(e['scores'][s])-e['scores'][s][a]
                        for group in GROUPS:
                            val=binary_cell(v-v[a],ev,candidate_mask(e['geometry'],a,group))
                            if val is not None:
                                binary.append(dict(**{k:base[k] for k in ['dataset','split','source_id','condition','order','arrival']},
                                    signal=s,group=group,**val))
                else:totals['nonexperts']+=1
                for n in [8,32]:
                    o=float(v[:n].max()) if r['expert_scheduled'] else r['A8_v'];base[f'O{n}_v']=o
                    for s in SIGNALS:
                        arm=f'{s}{n}';idx=e['choices'][arm] if r['expert_scheduled'] else None
                        sv=float(v[idx]) if idx is not None else r['A8_v'];st=float(t[idx]) if idx is not None else r['A8_t'];d=sv-r['A8_v']
                        base.update({arm+'_v':sv,arm+'_t':st,arm+'_gain':d,arm+'_t_gain':st-r['A8_t'],
                            arm+'_regret':o-sv,arm+'_gross_gain':max(d,0.),arm+'_gross_loss':max(-d,0.),
                            arm+'_changed':float(idx is not None and idx!=a),arm+'_severe':float(d<-.05),
                            arm+'_destroyed_old_fast':float(r['expert_scheduled'] and r['old_fast_gain']>EPS and d< -EPS)})
                rows.append(base)
            summaries={};binary_summaries={}
            fields=[k for k,vv in rows[0].items() if isinstance(vv,(int,float)) and not isinstance(vv,bool) and k not in ['source_id','arrival','anchor_index']]
            for gr in ['corruption','clean']:
                summaries[gr]={};binary_summaries[gr]={}
                for sub in ['all','expert','nonexpert']:
                    part=[r for r in rows if (r['condition']=='clean')==(gr=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                    az=aggregate(part,fields);az['transitions']={}
                    for s in SIGNALS:
                        for n in [8,32]:
                            arm=f'{s}{n}';az['transitions'][arm]=dict(improved=sum(r[arm+'_gain']>EPS for r in part),
                                harmed=sum(r[arm+'_gain']< -EPS for r in part),unchanged=sum(abs(r[arm+'_gain'])<=EPS for r in part),
                                severe_harm_gt5pp=sum(r[arm+'_gain']<-.05 for r in part),
                                correctness={str(q):dict(destroyed=sum(r['A8_v']>q>=r[arm+'_v'] for r in part),
                                    rescued=sum(r[arm+'_v']>q>=r['A8_v'] for r in part)) for q in [.3,.5]})
                    summaries[gr][sub]=az
                for group in GROUPS:
                    binary_summaries[gr][group]={}
                    for s in SIGNALS:
                        rr=[r for r in binary if r['group']==group and r['signal']==s and (r['condition']=='clean')==(gr=='clean')]
                        az=aggregate(rr,['tp','fp','fn','tn','positive','negative','accepted'],
                            dict(precision=('tp','accepted'),benefit_recall=('tp','positive'),harm_acceptance=('fp','negative')))
                        ar=[r for r in rr if r['auc'] is not None];az['auc']=aggregate(ar,['auc'])
                        az['raw_counts']={k:sum(r[k] for r in rr) for k in ['candidates','positives','negatives','TP','FP','FN','TN']}
                        az['eligible_auc_cells']=len(ar);az['undefined_auc_cells']=len(rr)-len(ar)
                        binary_summaries[gr][group][s]=az
            out=PUBLIC/sp/ds;write(out/'ROWS.json',rows);write(out/'SUMMARY.json',summaries)
            write(out/'BINARY_ROWS.json',binary);write(out/'DISCRIMINATION.json',binary_summaries)
            examples={}
            for s in SIGNALS:
                arm=s+'32';rr=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
                examples[s]=dict(positive=sorted(rr,key=lambda r:-r[arm+'_gain'])[:3],negative=sorted(rr,key=lambda r:r[arm+'_gain'])[:3])
            write(out/'CASES.json',examples);print('EVIDENCE_DIAGNOSIS_PANEL',ds,sp,len(rows),flush=True)
    assert dict(totals)==dict(arrivals=1152,experts=288,clean_experts=48,nonexperts=864,corrupt_experts=240)
    write(PUBLIC/'COVERAGE.json',dict(totals));write(PUBLIC/'LABEL_JOIN.json',dict(time=time.time(),
        signal_seal_time=seal['time'],signal_seal_sha256=sha(PUBLIC/'SIGNAL_SEAL.json'),
        labels_already_published=True,annotation_files_opened=0,inputs=label_files))
    write(PUBLIC/'RESOURCES.json',dict(signal=read(BASE/'SIGNAL_RESOURCES.json'),
        diagnosis_CPU_wall_seconds=time.time()-tick,GPU_used=False,new_model_calls=0,new_expert_calls=0,
        new_decoder_replays=0,new_backprop=0,new_inference_predictions=0,
        diagnostic_readouts=1152*6,wall_is_not_GPU_kernel_time=True,python=platform.python_version()))
    verify();status('completed_pending_independent_audit_report_publication',coverage=dict(totals))
    print('LARGE_EVIDENCE_CPU_COMPLETE',json.dumps(dict(totals)),flush=True)

if __name__=='__main__':{'prepare':prepare,'signals':signals,'diagnose':diagnose}[sys.argv[1]]()
