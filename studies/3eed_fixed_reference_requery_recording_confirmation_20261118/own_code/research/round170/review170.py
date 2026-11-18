"""Create output-blind subject/reference plates with shuffled annotation keys."""
from collections import Counter
import hashlib

import numpy as np
from PIL import Image

from scope170 import HERE, INPUT, SPEC, WORK, atomic_json, lock, read, sha, status
from calibrate168 import image_geometry, render_path
from mechanisms162 import annotation_inventory

REVIEW=WORK/'blind_review_PRIVATE'
SHEETS=REVIEW/'plates'
LIST=REVIEW/'BOX_KEYS_PRIVATE.json'
ADJUDICATION=REVIEW/'ADJUDICATION_PRIVATE.json'
STATUSES=('verified','ambiguous','unannotated','insufficient_visibility','unresolved_role')


def verify_predictions():
    meta=read(HERE/'PREDICTION_SEAL.json')
    assert meta['input_seal']==sha(HERE/'INPUT_SEAL.json') and not meta['target_identity_labels_read']
    assert not meta['FINAL_access'] and meta['r159_r158_pca_bit_exact']
    assert len(meta['files'])==meta['n']
    assert all(sha(WORK/'predictions'/name)==v for name,v in meta['files'].items())
    return meta


def prepare():
    design=lock('visual_review',('select170.py','predict170.py','review170.py'))
    meta=verify_predictions()
    assert not LIST.exists()
    status('blind_review',physical_frames=meta['physical_frames'],direction_records=meta['n'])
    SHEETS.mkdir(parents=True,exist_ok=True,mode=0o700)
    corpus=read(INPUT)
    assert len(corpus)==meta['physical_frames'] and sha(INPUT)==read(HERE/'INPUT_SEAL.json')['registry']
    aliases=[]
    files={}
    for entry in corpus:
        k=entry['case']
        row=entry['row']
        label=row['platform']
        names,boxes=annotation_inventory(row['metapath'],label,int(row['suffix'] or 0),row['object_index'])
        seed=hashlib.sha256((read(SPEC)['sampling_seed']+f'|blind-key|{k}').encode()).digest()
        perm=np.random.default_rng(int.from_bytes(seed[:8],'little')).permutation(len(names))
        role=entry['parsed']['graph']['entities'][entry['role']]
        lo,hi=role['span']
        term=row['caption'][lo:hi]
        with Image.open(row['imagepath']) as img:
            rgb=img.convert('RGB')
        render_path(k,row,image_geometry(row),[names[n] for n in perm],boxes[perm],
                    [f'Reference mention to check  {term}'],rgb,SHEETS)
        aliases.append(dict(case=k,display_to_annotation=perm.tolist(),inventory_size=len(names)))
        for name in (f'{k:02}.jpg',f'{k:02}_raw.jpg'):
            files[name]=sha(SHEETS/name)
        with Image.open(SHEETS/f'{k:02}.jpg') as img:
            img.thumbnail((990,860))
            img.save(SHEETS/f'{k:02}_compact.png')
    atomic_json(LIST,aliases)
    atomic_json(HERE/'REVIEW_PANEL_SEAL.json',dict(n=len(aliases),predictions=sha(HERE/'PREDICTION_SEAL.json'),
                design_lock=design,registry=sha(LIST),files=files,shuffled_box_labels=True,
                rendered_model_outputs=False,final_access=False))
    print('R170 prediction-blind sheets ready',len(aliases),flush=True)


def seal():
    design=lock('visual_review',('select170.py','predict170.py','review170.py'))
    check=read(HERE/'REVIEW_PANEL_SEAL.json')
    assert check['registry']==sha(LIST) and check['design_lock']==design
    assert check['predictions']==sha(HERE/'PREDICTION_SEAL.json') and not check['rendered_model_outputs']
    assert all(sha(SHEETS/name)==digest for name,digest in check['files'].items())
    items=read(LIST)
    judged=read(ADJUDICATION)
    assert [x['case'] for x in judged]==[x['case'] for x in items]
    states=Counter()
    for x,y in zip(items,judged,strict=True):
        s=x['inventory_size']
        for key in ('subject','reference'):
            found=y[key]
            assert found['status'] in STATUSES
            assert (0<=found.get('box',-1)<s)==(found['status']=='verified')
            states[key,found['status']]+=1
        if y['subject']['status']==y['reference']['status']=='verified':
            assert y['subject']['box']!=y['reference']['box']
    payload=dict(n=len(judged),reviewers=1,blinded_to_predictions_and_official_subject_slot=True,
                 all_rows_retained=True,certificate=sha(HERE/'REVIEW_PANEL_SEAL.json'),
                 adjudication=sha(ADJUDICATION),states={f'{k}:{v}':n for (k,v),n in sorted(states.items())},FINAL_access=False)
    atomic_json(HERE/'VISUAL_TRUTH_SEAL.json',payload)
    print(payload,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('prepare','seal'))
    {'prepare':prepare,'seal':seal}[p.parse_args().phase]()
