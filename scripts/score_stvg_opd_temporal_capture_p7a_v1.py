"""P7-A postseal full factorial scoring and independent source/dense/stat root."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *
SPATIAL=['Native','OPD','Direct','Capped_Direct']
PUB=OUT/'P7_A'

def array(x):return x.detach().cpu().numpy().astype(float)
def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b);return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b
    else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
    return 1

def rows_write(path,rows):
    assert not path.exists()
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
            for r in rows:f.write((json.dumps(r,allow_nan=False,separators=(',',':'))+'\n').encode())
def rows_read(path):
    with gzip.open(path,'rt') as f:return [json.loads(s) for s in f]

def require_seal():
    verify();b=read(A/'GLOBAL_READOUT_BARRIER.json');assert b['status']=='sealed' and b['cells']==512 and b['logical_outputs']==4096 and not b['GT_read']
    assert b['runtime_sha256']==sha(BASE/'RUNTIME.json');return b

def statistics(rows,subset,ds,order=None):
    rr=[r for r in rows if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order is None or r['order']==order)]
    by=collections.defaultdict(list)
    for r in rr:by[r['parent_id']].append(r)
    fields=[k for k,v in rr[0].items() if isinstance(v,(int,float)) and k not in ['parent_id','public_cell','previous_P6']]
    matrix=np.array([[np.mean([r[f] for r in rs]) for f in fields] for rs in by.values()])
    gen=np.random.default_rng(20261012);boot=[]
    for _ in range(100):boot.append(matrix[gen.integers(len(matrix),size=(100,len(matrix)))].mean(1))
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    return dict(cells=len(rr),parents=len(by),statistics={f:dict(mean=float(matrix[:,j].mean()),ci95=ci[:,j].tolist(),
        parent_values=matrix[:,j].tolist(),harm_5pp=int(sum(matrix[:,j]<-.05)),harm_20pp=int(sum(matrix[:,j]<-.2)),
        positive_parents=int(sum(matrix[:,j]>1e-12)),negative_parents=int(sum(matrix[:,j]< -1e-12)),min=float(matrix[:,j].min()),max=float(matrix[:,j].max())) for j,f in enumerate(fields)})

def all_stats(rows):
    return {ds:{subset:dict(paired_both_orders=statistics(rows,subset,ds),orders={o:statistics(rows,subset,ds,o) for o in ['order1','order2']})
        for subset in ['all128','previous_P6_32','added96']} for ds in ['hc2','vidstg']}

def context_truth():
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.score_stvg_opd_p1_v1 import truths
    gt={};spans={};d=read(BASE/'DESIGN_LOCK.json')
    for ds in ['hc2','vidstg']:
        gt[ds],spans[ds],_=truths(ds,dict(parents=[r['query_ordinal'] for r in d['captures'] if r['dataset']==ds]))
    return gt,spans

def score():
    import torch
    torch.set_num_threads(2);barrier=require_seal();gt,spans=context_truth()
    write(A/'GT_EXPOSURE.json',dict(status='actual_offline_after_all512_eightway_seal',barrier_sha256=sha(A/'GLOBAL_READOUT_BARRIER.json'),
        historical_exposure=True,GT_for_interval_or_observation_selection=False,time=time.time()))
    from scripts.run_decota_paper_main_v1 import read_row
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    rows=[];checks=0
    for r in read(BASE/'DESIGN_LOCK.json')['records']:
        p=A/'readouts'/f"{r['public_cell']:05}.pt";assert sha(p)==barrier['files'][str(p.relative_to(ROOT))];z=load(p)
        ds=r['dataset'];q=r['query_ordinal'];raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q]
        row={k:r[k] for k in ['public_cell','dataset','parent_id','order','previous_P6']}
        for arm,box in z['boxes'].items():
            for t,interval in z['intervals'].items():
                values=official(array(box),raw,truth,span,interval,ds);dense,fids,iou=dense_metric(array(box),raw,truth,span,interval,ds)
                checks+=near(values,dense)
                for m,v in values.items():row[f'{arm}_{t}_{m}']=v
                for threshold in [.3,.5]:row[f'{arm}_{t}_R{int(threshold*100)}']=float(values['v']>threshold)
            assert row[f'{arm}_T0_s']==row[f'{arm}_T1_s']
            for m in ['v','t','s','R30','R50']:row[f'delta_T_{arm}_{m}']=row[f'{arm}_T1_{m}']-row[f'{arm}_T0_{m}']
        for t in ['T0','T1']:
            assert len({row[f'{a}_{t}_t'] for a in SPATIAL})==1
            for comparator in ['Native','Direct','Capped_Direct']:
                for m in ['v','t','s','R30','R50']:row[f'OPD_minus_{comparator}_{t}_{m}']=row[f'OPD_{t}_{m}']-row[f'{comparator}_{t}_{m}']
        for m in ['v','t','s','R30','R50']:row[f'interaction_OPD_Native_{m}']=row[f'delta_T_OPD_{m}']-row[f'delta_T_Native_{m}']
        ids=np.array(sorted(truth));event=(ids>=span[0])&(ids<span[1]);sample=np.asarray(raw['frame_ids']);support=(ids>=sample[0])&(ids<=sample[-1])
        a,b=z['intervals']['T0'];c,e=z['intervals']['T1'];m0=(ids>=a)&(ids<b);m1=(ids>=c)&(ids<e)
        for k,mask in [('gained_event_frames',event&m1&~m0),('lost_event_frames',event&m0&~m1),
            ('new_event_supported_frames',event&m1&~m0&support),('new_event_unsupported_frames',event&m1&~m0&~support),
            ('T0_annotated_outside_event_frames',~event&m0),('T1_annotated_outside_event_frames',~event&m1)]:row[k]=int(mask.sum())
        # Physical event-exterior expansion is counted even when no GT boxes
        # exist there; it is duration, never invented localization evidence.
        g,h=span
        def outside(interval):
            start,end=interval;return end-start-max(0,min(end,h)-max(start,g))
        row['T0_outside_event_duration']=outside(z['intervals']['T0']);row['T1_outside_event_duration']=outside(z['intervals']['T1'])
        row['T0_event_coverage']=float((event&m0).sum()/event.sum());row['T1_event_coverage']=float((event&m1).sum()/event.sum());row['delta_event_coverage']=row['T1_event_coverage']-row['T0_event_coverage']
        rows.append(row);status(A/'CPU_STATUS.json',dict(status='actual_all512_postseal_scoring',done=len(rows),total=512,GT_read=True,CPU_only=True,time=time.time()))
    assert len(rows)==512;rows_write(PUB/'ROWS.jsonl.gz',rows);write(PUB/'SUMMARY.json',all_stats(rows))
    write(A/'CPU_COMPLETION.json',dict(status='pending_actual_root_PDF_view_publication_archive',cells=512,outputs=4096,parents=256,
        official_dense_checks=checks,files={str(f.relative_to(ROOT)):sha(f) for f in [PUB/'ROWS.jsonl.gz',PUB/'SUMMARY.json']},CPU_only=True,GT_read=True,time=time.time()))
    print('P7A_ALL512_SCORED',flush=True)

def independent_interval(z):
    from scripts.stvg_opd_p6_existing_common001 import config
    cfg=config(z['dataset']);c=z['capture'];iv=[];checks=0
    for zz,action,r in zip(c['zero']['logits'],c['zero']['actions'],c['records']):
        v=array(zz).reshape(-1,2);x=array(action).astype(np.float32).astype(float).reshape(-1);f=np.array(r['frame_ids']);ij=np.triu_indices(len(f),1)
        scores=v[ij[0],0]+v[ij[1],1];mx=max(scores);lp=scores-(mx+np.log(np.exp(scores-mx).sum()))
        center=np.median(x);mad=np.median(np.abs(x-center));u=(x-cfg.center_fraction*center)/(mad+cfg.epsilon)
        edges=np.r_[f[0],(f[:-1]+f[1:])/2,f[-1]+1];weights=np.diff(edges)/np.diff(edges).sum();pre=np.r_[0.,np.cumsum(weights*u)]
        cost=np.sum(weights*np.logaddexp(0,u))-(pre[ij[1]+1]-pre[ij[0]]);i=int(np.argmax(-cost+cfg.prior_weight*lp))
        iv.append([int(f[ij[0][i]]),int(f[ij[1][i]]+1)]);checks+=len(scores)+len(x)
    return [min(x[0] for x in iv),max(x[1] for x in iv)],checks

def root():
    import torch
    torch.set_num_threads(2);barrier=require_seal();assert read(A/'CPU_COMPLETION.json')['GT_read'];gt,spans=context_truth()
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    rows=rows_read(PUB/'ROWS.jsonl.gz');summary=read(PUB/'SUMMARY.json');projection_values=0;checks=0;hashes=0;coordinates=0;temporal_seen=set()
    for index,r in enumerate(read(BASE/'DESIGN_LOCK.json')['records']):
        p=A/'readouts'/f'{index:05}.pt';h=sha(p);assert h==barrier['files'][str(p.relative_to(ROOT))];z=load(p);row=rows[index]
        tp=ROOT/z['temporal_binding']['path'];assert sha(tp)==z['temporal_binding']['sha256'];tz=load(tp)
        if str(tp) not in temporal_seen:
            interval,n=independent_interval(tz);assert interval==z['intervals']['T1'];projection_values+=n;temporal_seen.add(str(tp))
        ip=ROOT/r['original_Full']['input_path'];assert sha(ip)==r['original_Full']['input_sha256'];inp=normalized(load(ip))
        assert inp['interval']==z['intervals']['T0'] and inp['frame_ids']==tz['capture']['frame_ids'] and inp['pixel_sha256']==tz['pixel_sha256']
        assert inp['source_model_state_sha256']==tz['capture']['source_model_state_sha256']
        assert torch.equal(inp['native_boxes'],z['boxes']['Native']);coordinates+=inp['native_boxes'].numel()
        for arm,b in z['spatial_bindings'].items():
            sf=ROOT/b['path'];assert sha(sf)==b['sha256'];fit=load(sf)['fit'];assert torch.equal(fit['final'],z['boxes'][arm]);coordinates+=fit['final'].numel()
            assert sha(sf)==b['sha256'];hashes+=2
        ds=r['dataset'];q=r['query_ordinal'];raw=read_row(ds,q)
        for arm,box in z['boxes'].items():
            for t,interval in z['intervals'].items():
                values,ids,iou=dense_metric(array(box),raw,gt[ds][q],spans[ds][q],interval,ds)
                for m,v in values.items():checks+=near(v,row[f'{arm}_{t}_{m}'])
                for th in [.3,.5]:checks+=near(float(values['v']>th),row[f'{arm}_{t}_R{int(th*100)}'])
        assert sha(p)==h and sha(tp)==z['temporal_binding']['sha256'];hashes+=4
        status(A/'ROOT_STATUS.json',dict(status='actual_source_capture_eightway_dense_root',done=index+1,total=512,CPU_only=True,GT_read=True,time=time.time()))
    assert len(temporal_seen)==256
    # Independently rebuild each parent vector and one complete resampling
    # index array, rather than call the producer's statistics implementation.
    for ds,data in summary.items():
        for subset,blocks in data.items():
            for label,old in [('both',blocks['paired_both_orders'])]+list(blocks['orders'].items()):
                selected=[r for r in rows if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (label=='both' or r['order']==label)]
                by=collections.defaultdict(list)
                for r in selected:by[r['parent_id']].append(r)
                assert len(by)==old['parents']=={'all128':128,'previous_P6_32':32,'added96':96}[subset]
                gen=np.random.default_rng(20261012);inds=gen.integers(len(by),size=(10000,len(by)))
                for f,v in old['statistics'].items():
                    x=np.array([math.fsum(r[f] for r in rs)/len(rs) for rs in by.values()]);boot=x[inds].sum(1)/len(x)
                    expected=dict(mean=math.fsum(x)/len(x),ci95=np.quantile(boot,[.025,.975]).tolist(),parent_values=x.tolist(),
                        harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),positive_parents=int(sum(x>1e-12)),negative_parents=int(sum(x< -1e-12)),min=float(min(x)),max=float(max(x)))
                    checks+=near(expected,v)
    for p,h in barrier['source_files'].items():assert sha(ROOT/p)==h;hashes+=1
    write(A/'ROOT_READBACK.json',dict(status='pass',scope='all256 exact temporal sources/all512 matched Full-Direct-cap-source inputs/eightway dense and all128/old32/new96/per-order parent statistics',
        cells=512,logical_outputs=4096,parents=256,unique_temporal_captures=256,projection_values=projection_values,
        scalar_checks=checks,coordinate_checks=coordinates,second_hash_checks=hashes,CPU_only=True,GT_read=True,new_spatial_fits=0,new_DINO_calls=0,time=time.time()))
    means={ds:{subset:summary[ds][subset]['paired_both_orders']['statistics']['delta_T_OPD_v'] for subset in ['all128','added96']} for ds in ['hc2','vidstg']}
    eligible=all(st['mean']>0 for x in means.values() for st in x.values())
    write(A/'P7_B_BRANCH_DECISION.json',dict(status='decided_after_complete_A_root',P7_B_eligible=eligible,
        rule=read(BASE/'DESIGN_LOCK.json')['branch_P7_B']['rule'],settings=means,root_readback_sha256=sha(A/'ROOT_READBACK.json'),
        no_parent_selection=True,no_parameter_tuning=True,conditional_one_B_only=True,GT_read=True,time=time.time()))
    print('P7A_ROOT_PASS',checks,hashes,'B_ELIGIBLE',eligible,flush=True)

if __name__=='__main__':{'score':score,'root':root}[sys.argv[1]]()
