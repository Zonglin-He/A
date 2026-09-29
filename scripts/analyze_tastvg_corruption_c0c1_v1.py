"""Post-seal, parent-paired anatomy and deterministic candidate-oracle readback."""
import sys,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_corruption_c0c1_v1 import OUT,verify,MEDIUM,CONDITIONS
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_tastvg_evidence_vulnerability_v1 import metrics as drift,arr

METRICS=['sIoU','tIoU','vIoU_corrected']


def stats(values):
    a=np.array([v for v in values if v is not None],float)
    if not len(a):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260929);b=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))


def group(ds,dt):
    ls,lt=-ds,-dt
    if max(ls,lt)<=.05:return 'robust_no_material_ST_loss'
    if ls>max(lt,0)+.05:return 'S_dominant'
    if lt>max(ls,0)+.05:return 'T_dominant'
    return 'joint_or_mixed'


def independent_candidates(x):
    # Independent exhaustive Cartesian enumeration; native FP32 logsoftmax is
    # retained so ties audit the registered decoder arithmetic, not float64 MAP.
    cc=x['candidates'];layers=x['layers'];records=x['records'];ids=x['frame_ids'];expected=[]
    for layer in layers[::-1]:
        if layer['indices'] not in expected and len(expected)<8:expected.append(layer['indices'])
    entries=[]
    for z in layers[-1]['logits']:
        n=z.shape[1];s=z[:,:,0].log_softmax(1)[0];e=z[:,:,1].log_softmax(1)[0];mat=(s[:,None]+e[None,:]).numpy();ij=np.transpose(np.triu_indices(n,1));ss=mat[ij[:,0],ij[:,1]];order=np.lexsort((ij[:,1],ij[:,0],-ss));entries.append((ss[order].astype(float),ij[order]))
    (a,ai),(b,bi)=entries;values=a[:,None]+b[None,:];order=np.argsort(-values.reshape(-1),kind='stable');visited=0
    for flat in order:
        if len(expected)>=8:break
        i,j=divmod(int(flat),len(b));start=min(records[0]['frame_ids'][ai[i,0]],records[1]['frame_ids'][bi[j,0]]);end=max(records[0]['frame_ids'][ai[i,1]],records[1]['frame_ids'][bi[j,1]])
        ij=[ids.index(start),ids.index(end)];visited+=1
        if ij not in expected:expected.append(ij)
    assert expected==[c['indices'] for c in cc['temporal']]
    boxes=[]
    for layer in layers[::-1]:
        if not any(np.array_equal(arr(layer['boxes']),a) for a in boxes):boxes.append(arr(layer['boxes']))
    assert len(boxes)==len(cc['spatial'])
    assert all(np.array_equal(a,arr(b['boxes'])) for a,b in zip(boxes,cc['spatial']))
    return dict(cartesian_pairs=int(values.size),visited_until_K=visited,spatial=len(boxes),temporal=len(expected))


def run():
    import ijson
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==224
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    assert sha(p['labels_path'])==p['labels_sha256'];keys={r['key'] for r in p['rows']};gt={}
    with open(p['labels_path'],'rb') as f:
        for k,z in ijson.kvitems(f,'',use_float=True):
            if k in keys:gt[k]={n:z[n] for n in ['interval','boxes','valid','event_mask','missing_event_frames']}
    assert set(gt)==keys;write(OUT/'GT_SUBSET.json',gt)
    write(OUT/'GT_EXPOSURE.json',dict(keys=sorted(keys),retained_records=32,container_streamed=True,container_sha256=p['labels_sha256'],GT_for_scoring_only=True,candidates_and_evidence_sealed_first=True,barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),historically_exposed=True,time=time.time()))
    c0=[];c1=[];audits=[];cell_count=0;metric_calls=0
    for r in p['rows']:
        key=r['key'];truth=gt[key];ids=r['frame_ids'];valid=np.array(truth['valid'],bool)
        assert len(ids)==len(valid)==len(truth['boxes']);assert valid.any();assert truth['event_mask']==[truth['interval'][0]<=f<truth['interval'][1] for f in ids]
        assert np.all(np.asarray(truth['boxes'])[valid,2:]>0)
        def score(boxes,indices):
            nonlocal metric_calls
            m,_=checked_score(boxes,truth,ids,indices);metric_calls+=1;return {k:m[k] for k in METRICS}
        clean=load(OUT/'capture/clean'/f"{r['ordinal']:03}.pt");clean_score=score(clean['native']['boxes'],clean['native']['indices'])
        for cond in CONDITIONS:
            x=load(OUT/'capture'/cond/f"{r['ordinal']:03}.pt");old=load(OUT/'baseline'/cond/f"{r['ordinal']:03}.pt");assert torch.equal(x['native']['boxes'],old['native']['boxes']);assert all(torch.equal(a,b) for a,b in zip(x['native']['logits'],old['native']['logits']));cell_count+=1
            base=score(x['native']['boxes'],x['native']['indices']);delta={m:base[m]-clean_score[m] for m in METRICS};mm=[]
            for a,b in zip(clean['evidence'],x['evidence']):
                d,_=drift(a,b)
                for stage in (1,2):
                    for branch in ('s','t'):
                        name=f'Q{branch}{stage}';d[name+'_relative_norm']=float(np.linalg.norm(arr(b[name])-arr(a[name]))/max(np.linalg.norm(arr(a[name])),1e-30))
                mm.append(d)
            summary={m:float(np.mean([d[m] for d in mm if d[m] is not None])) if any(d[m] is not None for d in mm) else None for m in mm[0]}
            c0.append(dict(key=key,ordinal=r['ordinal'],condition=cond,baseline_clean=clean_score,native=base,delta=delta,failure_group=group(delta['sIoU'],delta['tIoU']),evidence_offsets=mm,evidence=summary))
            if 'candidates' not in x:continue
            audit=independent_candidates(x);audits.append(dict(key=key,condition=cond,**audit));cc=x['candidates']
            ts=[dict(origin=t['origin'],metrics=score(x['native']['boxes'],t['indices'])) for t in cc['temporal']]
            ss=[dict(origin=s['origin'],metrics=score(s['boxes'],x['native']['indices'])) for s in cc['spatial']]
            ti=max(range(len(ts)),key=lambda i:ts[i]['metrics']['tIoU']);si=max(range(len(ss)),key=lambda i:ss[i]['metrics']['sIoU'])
            assert all(abs(ts[0]['metrics'][m]-base[m])<1e-12 and abs(ss[0]['metrics'][m]-base[m])<1e-12 for m in METRICS)
            arms=dict(native=base,T_oracle=ts[ti]['metrics'],S_oracle=ss[si]['metrics'],TS_oracle=score(cc['spatial'][si]['boxes'],cc['temporal'][ti]['indices']))
            layer_scores=[score(l['boxes'],l['indices']) for l in x['layers']]
            c1.append(dict(key=key,ordinal=r['ordinal'],condition=cond,arms=arms,temporal_candidates=ts,spatial_candidates=ss,selected_T=ti,selected_S=si,unique_T=len(ts),unique_S=len(ss),layer_only_temporal_gain=max(l['tIoU'] for l in layer_scores)-base['tIoU'],uniform_T_gain=float(np.mean([z['metrics']['tIoU'] for z in ts]))-base['tIoU'],uniform_S_gain=float(np.mean([z['metrics']['sIoU'] for z in ss]))-base['sIoU'],gain_T=arms['T_oracle']['tIoU']-base['tIoU'],gain_S=arms['S_oracle']['sIoU']-base['sIoU']))
        print('SCORED',r['ordinal']+1,32,key,flush=True)
    assert len(c0)==224 and len(c1)==64 and len(audits)==64
    anatomy={}
    for cond in CONDITIONS:
        rr=[r for r in c0 if r['condition']==cond];a=dict(n=32,absolute={m:stats([r['native'][m] for r in rr]) for m in METRICS},delta={m:stats([r['delta'][m] for r in rr]) for m in METRICS},failure_groups=dict(collections.Counter(r['failure_group'] for r in rr)),evidence={m:stats([r['evidence'][m] for r in rr]) for m in rr[0]['evidence']},tails={},retention={})
        for m in METRICS:
            good=[r for r in rr if r['baseline_clean'][m]>=.5];a['tails'][m]=dict(loss_gt5pp=sum(r['delta'][m]<-.05 for r in rr),gain_gt5pp=sum(r['delta'][m]>.05 for r in rr));a['retention'][m]=dict(clean_good=len(good),corrupted_good=sum(r['native'][m]>=.5 for r in good))
        anatomy[cond]=a
    support={};decisions={}
    for cond in MEDIUM+['corrupted_parent_macro']:
        rr=[r for r in c1 if r['condition']==cond] if cond!='corrupted_parent_macro' else [r for r in c1 if r['condition']!='clean']
        def macro(fn):
            values=collections.defaultdict(list)
            for r in rr:values[r['key']].append(fn(r))
            return [float(np.mean(values[k])) for k in sorted(values)]
        a=dict(parents=16,cells=len(rr),absolute={arm:{m:stats(macro(lambda r,arm=arm,m=m:r['arms'][arm][m])) for m in METRICS} for arm in ['native','T_oracle','S_oracle','TS_oracle']},delta={},tails={})
        for arm in ['T_oracle','S_oracle','TS_oracle']:
            a['delta'][arm]={m:stats(macro(lambda r,arm=arm,m=m:r['arms'][arm][m]-r['arms']['native'][m])) for m in METRICS}
            a['tails'][arm]={'v_loss_gt5pp_cells':sum(r['arms'][arm]['vIoU_corrected']-r['arms']['native']['vIoU_corrected']<-.05 for r in rr),'v_loss_gt5pp_parents_mean':sum(x<-.05 for x in macro(lambda r:r['arms'][arm]['vIoU_corrected']-r['arms']['native']['vIoU_corrected']))}
        for name in ['uniform_T_gain','uniform_S_gain','layer_only_temporal_gain','gain_T','gain_S','unique_T','unique_S']:a[name]=stats(macro(lambda r:r[name]))
        a['opportunity_gt5pp']={b:sum(x>.05 for x in macro(lambda r:r['gain_'+b])) for b in ['S','T']}
        a['chosen_origins']={b:dict(collections.Counter(r['spatial_candidates' if b=='S' else 'temporal_candidates'][r['selected_'+b]]['origin'] for r in rr)) for b in ['S','T']};support[cond]=a
        if cond=='corrupted_parent_macro':
            for b in ['S','T']:
                z=a['gain_'+b];decisions[b]=dict(mean_gain=z['mean'],ci95=z['ci95'],parents_gt5pp=a['opportunity_gt5pp'][b],candidate_support_gate=bool(z['mean']>=.02 and z['ci95'][0]>0 and a['opportunity_gt5pp'][b]>=4))
    examples=[]
    for cond in CONDITIONS[1:]:
        rr=[r for r in c0 if r['condition']==cond]
        for role,fn in [('largest_S_loss',lambda r:r['delta']['sIoU']),('largest_T_loss',lambda r:r['delta']['tIoU']),('stable_ST',lambda r:abs(r['delta']['sIoU'])+abs(r['delta']['tIoU']))]:
            z=min(rr,key=lambda r:(fn(r),r['ordinal']));examples.append(dict(role=role,**z))
    write(OUT/'analysis/C0_ROWS.json',c0);write(OUT/'analysis/C1_ROWS.json',c1);write(OUT/'analysis/C0_SUMMARY.json',anatomy);write(OUT/'analysis/C1_SUMMARY.json',support);write(OUT/'analysis/CASES.json',examples)
    write(OUT/'analysis/AUDIT.json',dict(status='pass',native_cells_exact=cell_count,candidate_sets=64,independent_candidate_enumerations=audits,dual_implementation_metric_calls=metric_calls,authorized_GT_keys=32,GT_generation=False,analysis_sha256=sha(Path(__file__))))
    write(OUT/'DECISION.json',dict(status='completed_audited',branches=decisions,gate_scope='finite resource gate for later critic qualification, not efficacy, no expert invocation now',corruption_damage_scope='paired C0 distinguishes actual corruption loss from preexisting clean error',C2='not_started',adaptation=False,HC2_used_for_selection=False,production_changed=False))
    print('C0/C1 complete',decisions)


if __name__=='__main__':run()
