"""Role-specific 3EED inference on unchanged physical A anchors. No task labels."""
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'research/round97'), str(ROOT/'crossplatform/round7')]
from core97 import base97, model97
from model51 import SENTINEL
from model76 import tensor_prefix
from camera_projection import CameraSpec, project_model_points, read_camera_allowlist

WORK = ROOT/'crossplatform/round158'
REG = ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json'
LOCK = HERE/'experiment/DESIGN_LOCK.json'
CORNERS = np.array(list(itertools.product((-.5,.5),repeat=3)))
EDGES = {'LEFT':-1, 'RIGHT':1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path, names):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in names}


def tensor(x):
    return torch.as_tensor(x, device='cuda')


def atomic_npz(path, **values):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as fp:
        np.savez_compressed(fp, **values)
    tmp.replace(path)


def role_plan(caption, graph):
    entities = {e['role']:e for e in graph['entities']}
    target = entities.get('T')
    def name(role):
        entity=entities.get(role,{})
        return 'T' if entity.get('head','').lower() in {'target','subject'} else role
    edges=set()
    for e in graph['relations']:
        if not e['known_frame'] or e['predicate'] not in EDGES:
            continue
        a,b = name(e['subject']),name(e['reference'])
        if a==b or 'T' not in (a,b):
            continue
        role=b if a=='T' else a
        if role in entities:
            edges.add((role,EDGES[e['predicate']]*(1 if a=='T' else -1)))
    # Opposite claims about the same anchor have no unique horizontal direction.
    edges=sorted((r,s) for r,s in edges if (r,-s) not in edges)
    items=[] if target is None else [('T',target)]+[(r,entities[r]) for r,s in edges]
    phrases=[]
    for role,e in items:
        phrase=e['phrase'].strip(' ;,.')
        if role=='T':
            match=re.search(r'\b(?:main )?appearance (?:features?|characteristics?)\s*:\s*([^;.]+)',caption,re.I)
            if match and match[1].strip().lower() not in phrase.lower():
                phrase+=', '+match[1].strip()
        phrases.append(phrase)
    return items,phrases,edges


def head_span(text, entity, local=False):
    if entity is None:
        return None
    lo,hi=(0,len(text)) if local else entity['span']
    if not 0<=lo<hi<=len(text):
        return None
    head=entity['head'].strip()
    m=re.search(r'\b'+re.escape(head)+r'(?:es|s)?\b',text[lo:hi],re.I) if head else None
    return (lo+m.start(),lo+m.end()) if m else (lo,hi)


def token_input(base,text,span,fallback=None):
    tok=base.tokenizer([text+SENTINEL],return_offsets_mapping=True,return_tensors='pt')
    off=tok.pop('offset_mapping')[0].numpy()
    mask=np.zeros(len(off),bool) if span is None else (off[:,1]>span[0])&(off[:,0]<span[1])&(off[:,1]>off[:,0])
    if not mask.any():
        mask=np.asarray(fallback,bool) if fallback is not None else (off[:,1]>off[:,0])&(off[:,1]<=len(text))
    if not mask.any() or mask.all() or len(mask)!=len(off):
        raise ValueError('invalid entity tokenization')
    return tok.to('cuda'),tensor(mask)


def logits(q,u,mask):
    scores=q@u.T/.07
    return (scores[:,mask].logsumexp(-1)-scores[:,~mask].logsumexp(-1)).double().cpu().numpy()


def rectangles(entry, boxes):
    row=entry['row']
    cam=CameraSpec.from_camera_fields(entry['target'],read_camera_allowlist(row['metapath']),
        waymo_view_suffix=row['suffix'],missing_distortion_policy='assume_zero_declared')
    with Image.open(entry['A_sensor']['imagepath']) as im:
        size=im.size
    xyz=boxes[:,None,:3]+CORNERS[None]*boxes[:,None,3:]
    proj=project_model_points(xyz.reshape(-1,3),cam,size)
    uv=proj['uv'].reshape(256,8,2)/size
    visible=np.isfinite(uv).all((1,2)) & (boxes[:,3:]>0).all(1)
    uv=np.nan_to_num(uv)
    low,high=uv.min(1),uv.max(1)
    visible &= (high>0).all(1)&(low<1).all(1)
    return np.stack([np.clip(low,0,1),np.clip(high,0,1)],1),visible


def data_graph(g, image):
    d={k:tensor(g[k]) for k in ('p2i','i2p','p2i_weight','i2p_weight')}
    d['image']=tensor(image.copy()).permute(2,0,1)[None].float()/255
    return d


def phrase_scores(base, m, z, graph, phrases, visual, lang):
    if not phrases:
        return np.empty((0,256))
    point, image, _=m.features(visual,graph,'bidirectional')
    ids=z['query_points_sample_inds'].long()[None]
    ik=ids[0]
    known=(graph['p2i'][ik]>=0).any(-1)
    late=torch.tanh(m.gate_late)*m.late(m.point_image(image,graph)[ik].float())*known[:,None]
    result=[]
    for text,mask in lang:
        pad=torch.zeros(text.shape[:2],device='cuda',dtype=torch.bool)
        v,l=base.cross_encoder(vis_feats=point[None],pos_feats=z['pos0'],
            padding_mask=torch.zeros((1,len(point)),device='cuda',dtype=torch.bool),
            text_feats=text,text_padding_mask=pad,end_points={},detected_feats=None,detected_mask=None)
        _,f,_=base.gsample_module(z['xyz'],v.transpose(1,2).contiguous(),ids.int())
        q=base.decoder_query_proj(f).transpose(1,2).contiguous()
        for i,decoder in enumerate(base.decoder):
            q=decoder(q,v,l,z['position'+str(i)],None,pad,detected_feats=None,detected_mask=None)
        q=F.normalize(m.match(q.float())[0]+late,dim=-1)
        u=F.normalize(base.contrastive_align_projection_text(l.float())[0],dim=-1)
        result.append(logits(q,u,mask))
    return np.stack(result)


def cached_full(m,z,prefix,mask):
    q=F.normalize(m.match(tensor(z[prefix+'_h'])[None])[0]+tensor(z[prefix+'_late']),dim=-1)
    return logits(q,tensor(z[prefix+'_u']),mask)


def extract(base,m,entry):
    row=entry['row']; sid=row['id']; direction=entry['source']+'_to_'+entry['target']
    if entry['cohort']!='EXT' or entry['protected_alias_fallback'] or not entry['A_sensor']['eligible_for_extra_media']:
        raise ValueError('input outside registered ordinary cohort')
    p131=ROOT/'crossplatform/round131/formal'/direction
    p135=ROOT/'crossplatform/round135/formal'/direction
    bi=ROOT/'crossplatform/round145/image'/direction
    pos=['position'+str(i) for i in range(6)]
    z0=arrays(p131/'prefix_PRIVATE'/(sid+'.npz'),['xyz','vis0','pos0','query_points_sample_inds','point_indices','boxes']+pos)
    z=tensor_prefix(z0); boxes=z0['boxes']; caption=row['caption']
    roles=json.loads((ROOT/'crossplatform/round97/roles_PRIVATE'/(sid+'.json')).read_text())
    if roles['caption_sha256']!=hashlib.sha256(caption.encode()).hexdigest():
        raise ValueError('caption mismatch')
    items,phrases,edges=role_plan(caption,roles['graph'])
    old=arrays(p135/'features_PRIVATE'/(sid+'.npz'),['intact_A_h','intact_A_late','intact_A_u','intact_A_mask','valid_B']+(['B_vis'] if entry['has_B'] else []))
    _,fullmask=token_input(base,caption,head_span(caption,items[0][1] if items else None),old['intact_A_mask'])
    a_full=cached_full(m,old,'intact_A',fullmask)
    lang=[]
    for phrase,(_,e) in zip(phrases,items):
        inp,mask=token_input(base,phrase,head_span(phrase,e,True))
        lang.append((base.text_projector(base.text_encoder(**inp).last_hidden_state),mask))
    k=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest()
    gp=ROOT/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+k[:20]+'.npz')
    if json.loads(gp.with_suffix('.json').read_text())['geometry_seed_and_sample_sha256']!=k:
        raise ValueError('point support mismatch')
    g=arrays(gp,['p2i','p2i_weight','i2p','i2p_weight','image'])
    aa=phrase_scores(base,m,z,data_graph(g,g['image']),phrases,z['vis0'][0],lang)
    af=np.vstack([a_full,aa]); bf=af.copy(); coverage=np.zeros(256)
    if bool(old['valid_B'].any())!=entry['has_B']:
        raise ValueError('B validity mismatch')
    if entry['has_B']:
        extra=entry['pair_PRIVATE']['B']
        if not extra['eligible_for_extra_media'] or extra['protected_image_content_alias'] or digest(extra['imagepath'])!=extra['image_sha256']:
            raise ValueError('invalid registered camera bank')
        bfe=arrays(bi/'features_PRIVATE'/(sid+'.npz'),['norm_Bimage_'+k for k in ['h','late','u']])
        b_full=cached_full(m,bfe,'norm_Bimage',fullmask)
        keys=['p2i','p2i_weight','i2p','i2p_weight','visible']
        bank=arrays(bi/'banks_PRIVATE'/(sid+'.npz'),['norm_Bimage_'+k for k in keys])
        bg={k:bank['norm_Bimage_'+k] for k in keys}
        with Image.open(extra['imagepath']) as im:
            image=np.asarray(im.convert('RGB').resize((384,256),Image.Resampling.BILINEAR))
        bp=arrays(bi/'position_PRIVATE'/(sid+'.npz'),['pos0','positions','physical_xyz'])
        if not np.array_equal(bp['physical_xyz'],z0['xyz']):
            raise ValueError('B anchors shifted')
        bz=dict(z,pos0=tensor(bp['pos0']))
        bz.update({k:tensor(bp['positions'][i]) for i,k in enumerate(pos)})
        xyz=z0['xyz'][0]
        inside=np.all(np.abs(xyz[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2,axis=-1)
        coverage=(inside & (old['valid_B']&bg['visible'])[None]).sum(1)/inside.sum(1).clip(1)
        vis=torch.where(tensor(old['valid_B'])[:,None],(z['vis0'][0]+tensor(old['B_vis']))/2,z['vis0'][0])
        bb=phrase_scores(base,m,bz,data_graph(bg,image),phrases,vis,lang)
        bf=np.vstack([b_full,bb])
    rect,seen=rectangles(entry,boxes); refsign=np.array([s for _,s in edges],int)
    if not np.isfinite(af).all() or not np.isfinite(bf).all():
        raise ValueError('nonfinite role evidence')
    return dict(A=af,B=bf,coverage=coverage,rect=rect,seen=seen,signs=refsign,boxes=boxes,box_sha=np.array(hashlib.sha256(boxes.tobytes()).hexdigest()))


@torch.inference_mode()
def run():
    import argparse,time
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    plan=json.loads(LOCK.read_text())
    if digest(REG)!=plan['registry_sha256']:
        raise ValueError('registry fingerprint')
    rows=json.loads(REG.read_text()); out=WORK/('engineering' if args.smoke else 'evidence_PRIVATE')
    out.mkdir(parents=True,exist_ok=True)
    for src in ['drone','quad','waymo']:
        ids=[i for i,r in enumerate(rows) if r['source']==src]
        if args.smoke:
            ids=[next(i for i in ids if rows[i]['has_B']),next(i for i in ids if not rows[i]['has_B'])]
        if all((out/f'{i:04}.npz').exists() for i in ids):
            continue
        base=base97(src);m,_=model97(base,src);m.route=None;m.use_interface=False
        for j,i in enumerate(ids):
            path=out/f'{i:04}.npz'
            if not path.exists():
                values=extract(base,m,rows[i]);atomic_npz(path,**values)
            if (j+1)%100==0 or args.smoke:
                print(src,j+1,'/',len(ids),flush=True)
        del base,m
        torch.cuda.empty_cache()
    print('all role evidence complete',flush=True)

if __name__=='__main__':
    run()
