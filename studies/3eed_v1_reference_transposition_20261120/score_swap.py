"""Post-seal official localization, physical-identity checks and all paired controls."""
import argparse
import csv
import gzip
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

from registry_swap import HERE, ROOT, WORK, OBS, PRED, OUT, PARENT, ORD, PANELS, SPEC, ordinary, key, read, digest, save
from posterior_swap import ARMS
from endpoint_score import arrays_to_owners, write_gz
from score_roles import anon, csv_file, inventory, measure, record_intervals, normalize, state, score_boxes, many_iou, sensor_truth, sensor_frame_box
from diagnose_roles import COHORTS, historical_records, historical_endpoints


def seals():
    p,o = read(PRED/'SEALED.json'),read(OBS/'SEALED.json')
    assert not p['labels'] and not o['labels'] and not p['FINAL_access'] and not o['FINAL_access']
    assert p['n']==o['n']==len(p['files']) and p['files'].keys()==o['files'].keys()
    assert p['protocol']==digest(SPEC) and p['observed']==digest(OBS/'SEALED.json')
    assert o['pin']['roster']==digest(WORK/'roster.json') and p['source_training']==p['target_updates']==0
    for code in (o['pin']['code'],p['code']):
        assert all(digest(HERE/name)==h for name,h in code.items())
    return p,o


@lru_cache(maxsize=6)
def baseline_direction(direction):
    folder = ORD/'audited_iou_PRIVATE'
    assert digest(folder/'SEALED.json')=='102d6a15614fb5f49f2132278235c3121c75a97330a4a4f238abdbdd0d68d10b'
    name = direction+'.npz'
    assert digest(folder/name)==read(folder/'SEALED.json')['files'][name]['sha256']
    with np.load(folder/name) as z:
        return z['ious'][:,0].copy()


def best(boxes,target):
    flat = boxes.copy()
    flat[:,6] = 0
    return float(max(many_iou(boxes,target).max(),many_iou(flat,target).max()))


def run(repo):
    p,o = seals()
    history = historical_records(repo)
    panels = {n:read(path) for n,path in PANELS.items()}
    with gzip.open(PARENT/'results/all_rows.csv.gz','rt') as f:
        old = {(r['source'],r['case']):float(r['iou']) for r in csv.DictReader(f) if r['arm']=='v1'}
    labels = {}
    data = []
    err = 0.
    reasons = Counter()
    for rec in read(WORK/'roster.json'):
        panel,idx,src = rec['panel'],rec['case'],rec['source']
        target = rec['frame']['target']
        name = key(rec)
        assert digest(OBS/name)==o['files'][name] and digest(PRED/name)==p['files'][name]
        ordinary_row = panel.startswith('ordinary_')
        entry = {} if ordinary_row else panels[panel][idx]
        row = ordinary()[target][idx] if ordinary_row else entry['row']
        view = int(row['suffix'] or 0)
        truth,txt = inventory(row['metapath'],target,view,int(row['object_index']))
        assert normalize(row['caption'])==txt
        if panel in COHORTS:
            if (panel,idx) not in labels:
                labels[panel,idx],_ = historical_endpoints(COHORTS[panel],entry,truth,history)
            subject,reference = labels[panel,idx]
            tag=f"{COHORTS[panel]}/{entry['case']:02d}_{src}.npz"
            path=ROOT/'crossplatform/round171/predictions'/tag
            assert digest(path)==history[0][tag]
            with np.load(path) as native:
                seeds=native['seeds']
        else:
            subject,reference = (0,None) if ordinary_row else (0,int(entry['ref_id']))
            seeds=None
        group = ('source_proxy' if src==target else 'cross_proxy') if panel=='source_proxy' else ('natural' if panel in COHORTS else panel)
        base = baseline_direction(src+'_to_'+target)[idx] if ordinary_row else old[src,anon(f'{panel}|{idx}')]
        with np.load(OBS/name) as z,np.load(PRED/name) as d:
            raw,ref,box=d['raw'],d['refs'],d['boxes']
            # Only distinct geometries need repeated high precision upright IoU scoring.
            uniq,inv = np.unique(np.vstack((raw,box)),axis=0,return_inverse=True)
            local,e1 = score_boxes(uniq,truth[0])
            sensor = np.stack([sensor_frame_box(b,target,view) for b in uniq])
            gt_sensor = sensor_truth(target,str(Path(row['metapath']).resolve()),row['caption'],row['suffix'])
            cross,e2 = score_boxes(sensor,gt_sensor)
            np.testing.assert_allclose(local,cross,atol=1e-8,rtol=0.)
            for t in (.25,.5):
                np.testing.assert_array_equal(local>t,cross>t)
            ri,fi = np.split(local[inv],2)
            assert abs(fi[0]-base)<1e-8,(panel,idx,src,fi[0],base)
            err=max(err,e1,e2,np.max(abs(local-cross)),abs(fi[0]-base))
            union=[z['a']]
            if 'd' in z:
                union.append(z['d'])
            if 'views' in z:
                union.extend(z['views'][1:])
            original=best(z['a'],truth[0])
            allmax=best(np.vstack(union),truth[0]) if len(union)>1 else original
            own=arrays_to_owners([raw,ref,box],truth,seeds)
            why=str(z['reason'])
            reasons[group,why]+=1
            anchor=dict(panel=panel,group=group,source=src,target=target,case=anon(f'{panel}|{target}|{idx}'),
                 frame=anon(str(Path(row['pointpath']).resolve())),recording=anon(str(Path(row['metapath']).resolve().parent.parent)),
                 active=int(d['applied']),reason=why,known_subject=int(subject is not None),known_reference=int(reference is not None),
                 a256=original,all_pool=allmax,affected=int(d['observed_changes']),stationary=int(d['station_count']),
                 max_log_contrast=float(d['max_abs_log_cross']),inverse_error=float(z['inverse_error']))
            for k,arm in enumerate(ARMS):
                ui,ur,uf=(v[k] for v in own['unique_iou'])
                fields=dict(**anchor,arm=arm,raw_iou=float(ri[k]),iou=float(fi[k]),raw_changed=int(not np.array_equal(raw[k],raw[0])),
                        shape_changed=int(1-many_iou(raw[k:k+1],box[k])[0]>1.e-8),
                        subject_official=state(ui,0),final_official=state(uf,0),
                        subject_iou=state(ui,subject),reference_iou=state(ur,reference),final_iou_owner=state(uf,subject),
                        subject_seed='NA',reference_seed='NA',final_seed='NA')
                if seeds is not None:
                    si,sr,sf=(v[k] for v in own['seed_ownership'])
                    fields.update(subject_seed=state(si,subject),reference_seed=state(sr,reference),final_seed=state(sf,subject))
                data.append(fields)
    OUT.mkdir(exist_ok=True)
    assert len(data)==p['n']*len(ARMS)
    write_gz(OUT/'all_rows.csv.gz',data)
    metrics,paired,duals,geom,intervals=[],[],[],[],[]
    for group in sorted({r['group'] for r in data}):
        buckets=defaultdict(lambda:defaultdict(list))
        for r in data:
            if r['group']==group:
                buckets[r['source'],r['target']][r['arm']].append(r)
        mats={}
        recs={}
        for direction,by in buckets.items():
            assert all([r['case'] for r in by[a]]==[r['case'] for r in by['v1']] for a in ARMS)
            mats[direction]=np.column_stack([[r['iou'] for r in by[a]] for a in ARMS])
            recs[direction]={'recording':np.array([r['recording'] for r in by['v1']])}
        expanded=[(s+'_to_'+t,by,mats[s,t],measure(mats[s,t])) for (s,t),by in buckets.items()]
        stacked={a:sum((by[a] for by in buckets.values()),[]) for a in ARMS}
        expanded.append(('macro',stacked,np.concatenate(list(mats.values())),np.mean([measure(m) for m in mats.values()],axis=0)))
        for direction,by,mat,table in expanded:
            for k,a in enumerate(ARMS):
                rows=by[a]
                info=dict(group=group,direction=direction,arm=a,n=len(rows))
                metrics.append(dict(**info,active=sum(r['active'] for r in rows),miou=table[k,2],acc25=table[k,0],acc50=table[k,1]))
                for rule in ('iou','seed'):
                    rr=[r for r in rows if r['known_subject'] and r['known_reference'] and (r['panel'] in COHORTS or rule=='iou')]
                    duals.append(dict(**info,rule=rule,both_known=len(rr),subject=sum(r['subject_'+rule]=='T' for r in rr),
                                    reference=sum(r['reference_'+rule]=='T' for r in rr),
                                    both=sum(r['subject_'+rule]==r['reference_'+rule]=='T' for r in rr)))
                same=[r for r in rows if r['subject_official']=='T']
                well=[r for r in rows if r['subject_seed']=='T']
                geom.append(dict(**info,correct_raw=len(same),changed=sum(r['shape_changed'] for r in rows),
                     rescue50=sum(r['iou']>.5>=r['raw_iou'] for r in same),harm50=sum(r['raw_iou']>.5>=r['iou'] for r in same),
                     beyond_a256=sum(r['iou']>.5>=r['a256'] and r['final_official']=='T' for r in same),
                     beyond_all=sum(r['iou']>.5>=r['all_pool'] and r['final_official']=='T' for r in same),
                     human_correct_raw=len(well),human_rescue50=sum(r['iou']>.5>=r['raw_iou'] for r in well),
                     human_harm50=sum(r['raw_iou']>.5>=r['iou'] for r in well),
                     human_beyond_all=sum(r['iou']>.5>=r['all_pool'] and r['final_seed']=='T' for r in well)))
                for c in ARMS:
                    if not (c=='v1' or a in ('crossed_raw','crossed_interval')):
                        continue
                    j=ARMS.index(c)
                    comp=by[c]
                    pair=dict(**info,control=c,delta25=table[k,0]-table[j,0],delta50=table[k,1]-table[j,1],delta_miou=table[k,2]-table[j,2])
                    for t in (25,50):
                        first,last=mat[:,j]>t/100,mat[:,k]>t/100
                        pair['rescues'+str(t)]=int((last & ~first).sum())
                        pair['harms'+str(t)]=int((first & ~last).sum())
                    for rule in ('iou','seed'):
                        valid=[(u,v) for u,v in zip(rows,comp) if u['known_subject'] and u['known_reference'] and (u['panel'] in COHORTS or rule=='iou')]
                        current=[u['subject_'+rule]==u['reference_'+rule]=='T' for u,_ in valid]
                        before=[v['subject_'+rule]==v['reference_'+rule]=='T' for _,v in valid]
                        pair[rule+'_dual_rescues']=sum(x and not y for x,y in zip(current,before))
                        pair[rule+'_dual_harms']=sum(y and not x for x,y in zip(current,before))
                    paired.append(pair)
        for a,c in (('crossed_interval','v1'),('crossed_raw','native_prior'),('crossed_raw','wrong_landmark'),('crossed_interval','interval_same_winner')):
            lo,hi=record_intervals(recs,mats,ARMS.index(a),ARMS.index(c))
            intervals.append(dict(group=group,arm=a,control=c,lo_acc50_pp=lo,hi_acc50_pp=hi))
    for name,records in (('metrics',metrics),('paired',paired),('identity',duals),('geometry',geom),('intervals',intervals)):
        csv_file(OUT/(name+'.csv'),records)
    save(OUT/'audit.json',dict(checked_baseline_rows=p['n'],rows=len(data),active=p['active'],observations=digest(OBS/'SEALED.json'),
         predictions=digest(PRED/'SEALED.json'),protocol=digest(SPEC),max_iou_error=err,
         reasons={g+'|'+why:n for (g,why),n in sorted(reasons.items())},
         source_training=0,target_updates=0,FINAL_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path)
    run(parser.parse_args().archive)
