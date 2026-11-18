"""Build the frozen panel from text and registered non-protected sensor identities."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import sys

import ijson

from scope170 import (CAL, HERE, INPUT, PLATFORMS, R159, R168, REC_INPUT, ROOT, SPEC, TRAIN_INDEX, WAYMO_INDEX, WORK,
                       atomic_json, lock, read, sha, status)
from calibrate168 import edges_of, ordinary_content
from panel169 import rewrite
from retrieval167 import direct_edges
from score165 import anonymizer

HORIZONTAL=re.compile(r'\b(?:left|right)\b',re.I)


def frame(row):
    return Path(row['metapath']).resolve()


def indices():
    ids={}
    fields={'id','meta_path','metapath','object_index'}
    for p in (TRAIN_INDEX,WAYMO_INDEX):
        with p.open('rb') as f:
            for prefix,event,value in ijson.parse(f):
                if (prefix,event)==('item','start_map'):
                    x={}
                elif prefix.startswith('item.') and prefix[5:] in fields and event in ('string','number'):
                    x[prefix[5:]]=value
                elif (prefix,event)==('item','end_map'):
                    ids[x['id']]=(Path(x.get('meta_path',x.get('metapath'))).resolve(),int(x['object_index']))
    return ids


def parser():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'research/round97'))
    import parse97
    assets=ROOT/'crossplatform/round20/assets'
    cfg=read(assets/'STANZA_ASSET_RECEIPT.json')['kwargs']
    nlp=parse97.stanza.Pipeline('en',dir=str(assets/'stanza'),processors='tokenize,mwt,pos,lemma,depparse',
                    package=None,use_gpu=False,download_method=None,verbose=False,**cfg)

    def inspect(caption):
        doc=parse97.dependency_document(nlp(caption))
        subject=parse97.parse_subject(caption,doc)
        graph=parse97.repair_inventory87(caption,doc,parse97.parse85(caption,doc,subject))
        return dict(graph=graph,doc=doc,caption_sha256=hashlib.sha256(caption.encode()).hexdigest())
    return inspect


@lru_cache(maxsize=1)
def previously_reviewed():
    official=CAL/'ROSTER.json'
    assert sha(official)==read(R168/'SOURCE_INPUT_SEAL.json')['source_roster_sha256']
    source=[x['row'] for x in read(official)]
    return ({frame(x) for x in source},{sha(x['pointpath']) for x in source},{sha(x['imagepath']) for x in source})


def prepare():
    pinned=lock('selection',('select170.py',))
    status('selecting')
    rule=read(SPEC)
    seed=rule['sampling_seed']
    proof=read(R159/'RECORDING_INPUT_SEAL.json')
    assert proof['input_sha256']==sha(REC_INPUT) and not proof['target_labels_used_for_selection'] and not proof['FINAL_access']
    r159=read(REC_INPUT)
    assert len(r159)==1536
    old_pts,old_img,old_frames,old_recordings=ordinary_content()
    q_frames,q_points,q_images=previously_reviewed()
    index=indices()
    inspect=parser()
    partition=defaultdict(lambda:defaultdict(list))
    statistics=defaultdict(Counter)
    for i,r in enumerate(r159):
        platform=r['platform']
        count=statistics[platform]
        count['registered']+=1
        site=frame(r)
        record=site.parent.parent
        ph,ih=r['point_sha256'],r['image_sha256']
        if record in old_recordings or site in old_frames or ph in old_pts or ih in old_img:
            count['selection_dataset_overlap']+=1
            continue
        if site in q_frames or ph in q_points or ih in q_images:
            count['R168_source_quality_overlap']+=1
            continue
        if not HORIZONTAL.search(r['caption']):
            count['no_horizontal_word']+=1
            continue
        row=dict(r,seed=0,caption=r['caption'].strip())
        d=inspect(row['caption'])
        e=direct_edges(edges_of(d['graph']))
        if not e:
            count['no_direct_edge']+=1
            continue
        def key(role,sign):
            return hashlib.sha256(f"{seed}|edge|{row['id']}|{role}|{sign}".encode()).digest()
        role,sign=min(e,key=lambda x:key(*x))
        decision=rewrite(row['caption'],d,e,role,sign)
        loc,anno=index[r['id']]
        assert loc==site
        row['object_index']=anno
        rank=hashlib.sha256(f"{seed}|roster|{row['id']}|{ph}".encode()).digest()
        partition[platform][record].append(dict(row=row,baseline_index=i,parsed=d,role=role,sign=sign,
                                                  rewrite=decision,stable_rank=rank.hex()))
        count['text_eligible']+=1
        count['can_rewrite']+=int(decision['status']=='supported')
    kept=[]
    scans=set()
    images=set()
    frames=set()
    alias=anonymizer()
    for platform in PLATFORMS:
        groups=partition[platform]
        seen=Counter()
        for items in groups.values():
            items.sort(key=lambda x:x['stable_rank'])
        while sum(seen.values())<rule['frames_per_target']:
            ready=[x for x in groups if groups[x]]
            if not ready:
                raise RuntimeError('Insufficient distinct relationship frames in ordinary TRAIN on '+platform)
            record=min(ready,key=lambda g:(seen[g],groups[g][0]['stable_rank']))
            entry=groups[record].pop(0)
            row=entry['row']
            physical=frame(row)
            ph,ih=row['point_sha256'],row['image_sha256']
            if physical in frames or ph in scans or ih in images:
                continue
            assert sha(row['pointpath'])==ph and sha(row['imagepath'])==ih
            frames.add(physical)
            scans.add(ph)
            images.add(ih)
            seen[record]+=1
            entry.update(case=len(kept)+1, target=platform, recording=alias('rec',record.as_posix()))
            kept.append(entry)
    expected=rule['frames_per_target']*len(PLATFORMS)
    assert len(kept)==len(scans)==len(images)==len(frames)==expected
    assert not (frames&old_frames or scans&old_pts or images&old_img or frames&q_frames or scans&q_points or images&q_images)
    assert not ({frame(e['row']).parent.parent for e in kept}&old_recordings)
    WORK.mkdir(parents=True,exist_ok=True,mode=0o700)
    assert not INPUT.exists()
    atomic_json(INPUT,kept)
    meta=dict(n=expected,direction_n=expected*2,selection_lock=pinned,registry=sha(INPUT),
              historical_ordinary_TRAIN=True,recordings_overlap_with_D_selection=0,
              sensor_overlap_with_R165_or_R168_source=0,selected_by_prediction_or_annotation=False,
              counts_by_platform={t:dict(eligible=dict(statistics[t]),frames=sum(x['target']==t for x in kept),
                                       recordings=len({x['recording'] for x in kept if x['target']==t}),
                                       rewrites=dict(Counter(x['rewrite']['status'] for x in kept if x['target']==t)))
                                  for t in PLATFORMS},FINAL_access=False)
    atomic_json(HERE/'INPUT_SEAL.json',meta)
    print(meta,flush=True)


if __name__=='__main__':
    prepare()
