"""Score sealed predictions; keep the full population, paired failures and identity controls."""
import csv
import gzip
from collections import defaultdict
import numpy as np

from common import ARMS, CFG, OUT, WORK, catalog, prior, read, digest, row_id, pins, save
from scope183 import ordinary
from score_roles import score_boxes, sensor_frame_box, sensor_truth, csv_file
from endpoint_score import arrays_to_owners, write_gz
from shared_paths import anonymous
from terrain_followup import boxes_in_model

CONTROLS=tuple(CFG['reference_controls'])
NAMES=ARMS+CONTROLS


def changes(boxes,ref):
    equal=np.isclose(boxes,ref,rtol=0,atol=1e-7,equal_nan=True)
    return ~equal.all(-1)


def summarize(mat,flags):
    hit25=mat>.25
    hit50=mat>.50
    return (np.array([hit25.mean(0),hit50.mean(0),mat.mean(0)])*100,
            np.array([(hit25&~hit25[:,:1]).sum(0),(~hit25&hit25[:,:1]).sum(0),
                      (hit50&~hit50[:,:1]).sum(0),(~hit50&hit50[:,:1]).sum(0),
                      (mat>mat[:,:1]+1e-8).sum(0),(mat<mat[:,:1]-1e-8).sum(0),flags.sum(0)]))


def recording_interval(panels):
    rng=np.random.default_rng(7462)
    samples=[]
    for tgt in sorted({k[1] for k in panels}):
        sources=sorted(k for k in panels if k[1]==tgt)
        labels=None
        for key in sources:
            mat,rec=panels[key]
            names=sorted(set(rec))
            if labels is None:
                labels=names
                draws=rng.integers(len(names),size=(4000,len(names)))
            assert names==labels
            delta=(mat>.50).astype(int)-(mat[:,:1]>.50)
            sums=np.stack([delta[rec==n].sum(0) for n in names])
            nums=np.array([sum(rec==n) for n in names])
            samples.append(sums[draws].sum(1)/nums[draws].sum(1)[:,None])
    return np.percentile(100*np.mean(samples,axis=0),[2.5,97.5],axis=0)


def run():
    seal=read(WORK/'SEALED.json')
    assert seal['code']==pins() and not seal['labels_used']
    cat=catalog()
    expected={row_id(r) for r in cat['rows']}
    assert expected==set(seal['files'])
    for key,h in seal['files'].items():assert digest(WORK/'boxes'/(key+'.npz'))==h
    index={(r['source'],r['target'],r['ordinal']):r for r in cat['rows'] if r['scope']=='ordinary'}
    table=ordinary()
    past=prior.old
    base_arm=past.ARMS.index('shared_cut')
    root=prior.FULL/'predictions'
    dirs=read(root/'SEALED.json')['dirs']
    scores={};event={};registered={};recids={};detail=[];metrics=[];old={}
    with gzip.open(past.OUT/'full_task_rows.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            if r['arm']=='shared_cut':old[r['source'],r['case']]=r
    err=0.;baseline_err=0.
    for direction,checksum in dirs.items():
        src,tgt=direction.split('_to_');folder=root/direction
        assert digest(folder/'SEALED.json')==checksum
        assert digest(folder/'all.npz')==read(folder/'SEALED.json')['all']
        with np.load(folder/'all.npz') as full:
            original=full['measured'][:,0];after=full['completed'];app=full['registered']
        n=len(app);mat=np.empty((n,len(NAMES)))
        change=np.zeros(mat.shape,bool);registrations=[];records=[]
        for ix,row in enumerate(table[tgt]):
            case=anonymous((tgt,ix));ref=old[src,case]
            truth=sensor_truth(tgt,row['metapath'],row['caption'],row['suffix'])
            proposals=np.repeat(after[ix,base_arm,None,:],len(NAMES),axis=0)
            proposals[ARMS.index('no_fallback_ground')]=original[ix]
            for j,name in enumerate(CONTROLS,len(ARMS)):
                if name=='slim_v1':proposals[j]=original[ix]
                elif name=='terrain_v1':proposals[j]=after[ix,0]
                else:proposals[j]=after[ix,past.ARMS.index(name)]
            if app[ix]:
                uid=row_id(index[src,tgt,ix])
                with np.load(WORK/'boxes'/(uid+'.npz')) as z:
                    proposals[:len(ARMS)]=[sensor_frame_box(v,tgt,row['suffix']) for v in z['subject']]
                np.testing.assert_allclose(proposals[0,:6],after[ix,base_arm,:6],atol=1e-6,rtol=0)
                assert np.isclose(np.sin(proposals[0,6]-after[ix,base_arm,6]),0,atol=1e-7)
            vals,gap=score_boxes(proposals,truth)
            err=max(err,gap)
            baseline_err=max(baseline_err,abs(vals[0]-float(ref['completed'])),abs(vals[ARMS.index('no_fallback_ground')]-float(ref['v1'])) if not app[ix] else 0.)
            assert baseline_err<1.e-7
            mat[ix]=vals;change[ix]=changes(proposals,proposals[0])
            record=ref['recording'];records.append(record);registrations.append(app[ix])
            detail.append(dict(source=src,target=tgt,case=case,recording=record,frame=ref['frame'],active=int(app[ix]),
                               **dict(zip(NAMES,vals))))
        scores[src,tgt]=mat;event[src,tgt]=change;registered[src,tgt]=np.array(registrations);recids[src,tgt]=np.array(records)
        print('scored',direction,n,flush=True)
    for scope in ('all','eligible','fallback'):
        numbers=[];counts=[];n_total=0;panels={}
        def emit(d,n,num,c,ci=None):
            for k,a in enumerate(NAMES):
                row=dict(subset=scope,direction=d,n=n,arm=a,acc25=num[0,k],acc50=num[1,k],miou=num[2,k],
                         delta25=num[0,k]-num[0,0],delta50=num[1,k]-num[1,0],delta_miou=num[2,k]-num[2,0],
                         rescue25=c[0,k],harm25=c[1,k],rescue50=c[2,k],harm50=c[3,k],
                         improved=c[4,k],worsened=c[5,k],changed_subject_box=c[6,k],
                         ci50_low=ci[0,k] if ci is not None else '',ci50_high=ci[1,k] if ci is not None else '')
                metrics.append(row)
        for key,mat in scores.items():
            mask={'all':np.ones(len(mat),bool),'eligible':registered[key],'fallback':~registered[key]}[scope]
            sub=mat[mask];n,c=summarize(sub,event[key][mask])
            panels[key]=sub,recids[key][mask]
            emit('_to_'.join(key),len(sub),n,c)
            numbers.append(n);counts.append(c);n_total+=len(sub)
        emit('macro',n_total,np.mean(numbers,axis=0),np.sum(counts,axis=0),recording_interval(panels))
    csv_file(OUT/'metrics.csv',metrics)
    write_gz(OUT/'ordinary_rows.csv.gz',detail)
    # Directly count endpoint changes in experiments designed to trigger the exact solver.
    endpoints=defaultdict(lambda:np.zeros((4,len(ARMS)),int));noun_rows=[];pairs=defaultdict(list)
    labels=read(prior.EXTRA/'labels.json')
    assert digest(prior.EXTRA/'labels.json')==read(past.OUT/'noun_panel.json')['labels']
    for r in cat['rows']:
        path=WORK/'boxes'/(row_id(r)+'.npz')
        with np.load(path) as z:
            s=z['subject'];ref=z['reference'];complex_form=bool(z['complex_form'])
            key=r['scope'],'ordered' if complex_form else 'binary_or_fallback'
            count=endpoints[key]
            count[0]+=1;count[1]+=changes(s,s[0]);count[2]+=changes(ref,ref[0]);count[3]+=np.any(z['selected']!=z['selected'][0],axis=1)
            if r['scope']!='noun':continue
            e=labels[r['uid']];truth=np.asarray(e['truth'])
            with np.load(r['prior']) as a:own=arrays_to_owners([s,ref],truth,a['seeds'])
            io,_=score_boxes(s,truth[e['subject']])
            for rule,(sub,rf) in own.items():
                hits=np.vstack((sub==e['subject'],rf==e['reference']))
                pairs[r['source'],e['pair'],rule].append(hits)
                for j,name in enumerate(ARMS):
                    noun_rows.append(dict(source=r['source'],target=r['target'],case=r['uid'],pair=e['pair'],half=e['half'],
                                          arm=name,rule=rule,iou=io[j],subject=int(hits[0,j]),ref=int(hits[1,j])))
    csv_file(OUT/'endpoint_changes.csv',[dict(panel=k[0],form=k[1],arm=a,n=v[0,j],subject=v[1,j],reference=v[2,j],node_pair=v[3,j])
                                        for k,v in endpoints.items() for j,a in enumerate(ARMS)])
    nr=[]
    for rule in ('unique_iou','seed_ownership'):
        values=[np.logical_and(*v) for k,v in pairs.items() if k[2]==rule and len(v)==2];assert len(values)==64
        x=np.stack(values)
        for j,a in enumerate(ARMS):
            nr.append(dict(rule=rule,arm=a,n=64,both_subjects=x[:,0,j].sum(),both_references=x[:,1,j].sum(),
                           all_four=np.all(x[:,:,j],axis=1).sum()))
    csv_file(OUT/'noun_pairs.csv',nr);write_gz(OUT/'noun_rows.csv.gz',noun_rows)
    save(OUT/'audit.json',dict(records=len(detail),directions=len(dirs),arms=len(NAMES),new_configurations=len(ARMS),
         parsed=sum(x.sum().item() for x in registered.values()),predictions_sha256=digest(WORK/'SEALED.json'),
         protocol_sha256=digest(prior.ROOT/'studies'/OUT.parent.name/'PROTOCOL.json'),replay_iou_error=baseline_err,
         two_iou_implementations_max_error=err,reference_and_subject_model_replay=seal['errors'][0],
         pair_factor_error=seal['errors'][1],observation_only_cpu_sweep_seconds=seal['seconds'],
         controlled_noun_pairs=64,all_prior_outputs_checked=2626,FINAL_access=False))
    print('complete',len(metrics),len(detail),flush=True)


if __name__=='__main__':
    run()
