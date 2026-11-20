"""New recording-separated natural Waymo pairs; all decisions precede identity scoring."""
from collections import defaultdict
from time import perf_counter
from bias import (DUEL, HERE, RESULTS, WORK, anon, archive_npz, atomic, digest, initialize,
                  loaded, np, predict, read, write_csv)
from common import PREV, engine, legacy, torch
from fit_duels import all_scores, versions
from scope import canonical, file_sha, physical_key, source_rows
from scope183 import ordinary, frame
from source_features import common_nominations, native_boxes, ownership, reencode
from source_pairs import catalog,source_identity


def choose():
    previous=[r for s in ('fit','calibration') for r in source_rows('waymo',s)]
    used_rec={r['recording'] for r in previous}
    used_cloud={r['point_sha256'] for r in previous}
    by_file=defaultdict(list)
    for r in ordinary()['waymo']:
        if r['recording'] not in used_rec:
            by_file[r['metapath']].append(r)
    choices=defaultdict(list)
    for rows in by_file.values():
        if len(rows)<2:
            continue
        base=rows[0]
        objs,pairs=catalog(base)
        available={r['object_index']:r for r in rows}
        for a,b in pairs:
            if a not in available or b not in available:
                continue
            physical=physical_key('waymo',base['metapath'])
            rank=anon('bias-confirm-v1',*physical,base['suffix'],a,b)
            choices[base['recording']].append((rank,physical,available[a],a,b))
    seen=set()
    roster=[]
    for rec in sorted(choices):
        count=0
        for key,physical,row,a,b in sorted(choices[rec],key=lambda z:z[0]):
            if physical in seen:
                continue
            h=file_sha(row['pointpath'])
            if h in used_cloud:
                continue
            used_cloud.add(h)
            seen.add(physical)
            roster.append(dict(key=key,row=row,a=a,b=b,point_hash=h))
            count+=1
            if count==2:
                break
    assert all(r['row']['recording'] not in used_rec for r in roster)
    return roster


def predictions(h,j):
    lock=read(WORK/'LOCK.json')
    assert lock['model']==digest(WORK/'model.npz') and lock['code']==digest(HERE/'bias.py')
    result=predict(h,j,loaded(WORK/'model.npz'),lock['scales'])
    if len(h)==0:
        return result
    prev=DUEL/'run_PRIVATE'
    other=read(prev/'DECODER_LOCK.json')
    assert digest(prev/'heads.npz')==read(prev/'HEADS_LOCK.json')['heads']
    heads=torch.from_numpy(loaded(prev/'heads.npz')['heads'])
    others=versions(all_scores(h,j,heads[0],heads),other['scales'])
    for arm in ('independent','worst','assignment','broken_pair','old_mean','old_worst'):
        result['duel_'+arm]=others[arm]
    return result


@torch.inference_mode()
def capture_new():
    initialize()
    assert read(WORK/'LOCK.json')['config']==digest(HERE/'PROTOCOL.json')
    roster=choose()
    atomic(WORK/'fresh_roster.json',roster)
    model=engine()
    hs,js,boxes=[],[],[]
    start=perf_counter()
    for i,item in enumerate(roster):
        r,a,b=item['row'],item['a'],item['b']
        objs,pairs=catalog(r)
        assert (a,b) in pairs
        f=frame(r)
        # Both queries use beam/packet support from exactly the same sampling.
        views={idx:model._read_observation(dict(f,text=canonical(objs[idx]['caption']))) for idx in (a,b)}
        np.testing.assert_array_equal(views[a][0].sampled_points,views[b][0].sampled_points)
        left,right=sorted((a,b),key=lambda idx:canonical(objs[idx]['caption']))
        bank=common_nominations(native_boxes(views[left][1]),native_boxes(views[right][1]))
        obs=reencode(model,bank,views,(a,b))
        hs.append(obs['h'])
        js.append(obs['j'])
        boxes.append(bank)
        if (i+1)%25==0:
            print('fresh source',i+1,len(roster),round(perf_counter()-start,1),flush=True)
    h,j=np.stack(hs),np.stack(js)
    arms=predictions(h,j)
    archive_npz(WORK/'fresh_predictions.npz',h=h,j=j,boxes=np.stack(boxes),
                logits=np.stack(list(arms.values())),arms=np.array(list(arms)))
    atomic(WORK/'FRESH_SEAL.json',dict(predictions=digest(WORK/'fresh_predictions.npz'),
        roster=digest(WORK/'fresh_roster.json'),lock=digest(WORK/'LOCK.json'),code=digest(HERE/'fresh_source.py'),
        count=len(h),recordings=len({r['row']['recording'] for r in roster}),elapsed=perf_counter()-start,
        identity_scoring_after_seal=True,labels_for_nomination=False,final_access=False))


def scored_rows(split,probe,keys,recs,slots,hits,arms):
    rows=[]
    for i,key in enumerate(keys):
        a,b=slots[i]
        available=a>=0 and b>=0
        for arm,arr in arms.items():
            s=arr[i]
            margins=[s[0,a]-s[0,b],s[1,b]-s[1,a]] if available else None
            choices=s.argmax(-1)
            both_four=hits[i,0,choices[0]] and hits[i,1,choices[1]]
            rows.append(dict(split=split,probe=probe,pair=str(key)[:16],rec_group=str(recs[i])[:12],arm=arm,
                available=int(available),both=int(available and min(margins)>0),both_four=int(both_four),
                m_a=float(margins[0]) if available else '',m_b=float(margins[1]) if available else ''))
    return rows


def score():
    seal=read(WORK/'FRESH_SEAL.json')
    assert seal['predictions']==digest(WORK/'fresh_predictions.npz') and seal['roster']==digest(WORK/'fresh_roster.json')
    roster=read(WORK/'fresh_roster.json')
    data=loaded(WORK/'fresh_predictions.npz')
    hits,slots=[],[]
    for i,r in enumerate(roster):
        row=r['row']
        objs,_=catalog(row)
        matrix,inv=source_identity(row,objs,data['boxes'][i])
        match,idx=ownership(matrix,inv[[r['a'],r['b']]])
        hits.append(match)
        slots.append(idx)
    rows=scored_rows('fresh','two_nominations',[r['key'] for r in roster],
        [anon('rec',r['row']['recording']) for r in roster],np.stack(slots),np.stack(hits),
        dict(zip(data['arms'],data['logits'])))
    for split in ('fit','calibration'):
        z=loaded(DUEL/'run_PRIVATE'/(split+'.npz'))
        for probe,p in [('pretruncation',''),('two_nominations','free_')]:
            scores=predictions(z[p+'h'],z[p+'j'])
            rows+=scored_rows(split,probe,z['key'],z['rec_key'],z[p+'slots'],z[p+'hits'],scores)
    from score import compressed_csv
    compressed_csv(RESULTS/'source_all.csv.gz',rows)
    bins=defaultdict(list)
    for r in rows:
        bins[r['split'],r['probe'],r['arm']].append(r)
    summary=[]
    for (split,probe,arm),group in bins.items():
        frozen={r['pair']:r for r in bins[split,probe,'v1']}
        eligible=[r for r in group if r['available']]
        match=sum(r['m_a']+r['m_b']>0 for r in eligible)
        summary.append(dict(split=split,probe=probe,arm=arm,n=len(group),recordings=len({r['rec_group'] for r in group}),
            covered=len(eligible),both=sum(r['both'] for r in group),both_four=sum(r['both_four'] for r in group),
            true_assignment=match,rescue=sum(r['both']>frozen[r['pair']]['both'] for r in group),
            harm=sum(r['both']<frozen[r['pair']]['both'] for r in group)))
    write_csv(RESULTS/'source_summary.csv',summary)


if __name__=='__main__':
    capture_new()
    score()
