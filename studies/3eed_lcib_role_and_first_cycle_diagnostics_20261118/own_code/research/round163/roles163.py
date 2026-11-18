"""Recover all 33 R91 shared assignments and preserve the original spans and role stages."""
from collections import Counter
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scope163 import (HERE, PARENT, PARENT_WORK, ROOT, SETTINGS, WORK, atomic_json, read,
                      sensor_row, sha, target_rows)
from geometry162 import CameraX, in_box
from mechanisms162 import annotation_inventory, find_expected_ref, r91_evidence, read_evidence, seed_owners
from mechanisms162 import support_owner
from score162 import table
from solver162 import Options, Relationships, build_roles, choose
from support162 import SceneSupport
from visual162 import LINES, corners, projection

STAGES=('native_full','native_top16','r88_region','r91_init','lcib_init','lcib_final')


def top16(affinity):
    w=np.zeros_like(affinity)
    keep=np.argsort(-affinity,kind='stable')[:16]
    w[keep]=affinity[keep]
    return w


def role_stages(ev,p,source,known,n_obj,old):
    scene=SceneSupport(ev)
    opts=Options(**source['options'])
    roles=build_roles(scene,opts,np.asarray(source['log_dimensions']),np.asarray(source['log_scale']))
    rel=Relationships(ev,roles,opts.relation_weight)
    selection=choose(scene,roles,rel)
    answer=[]
    for role in range(len(ev.affinity)):
        old_inside=in_box(ev.seeds,old)
        r88=ev.supports[role,int(np.argmax(ev.route_weights[role]))]
        init=np.zeros(len(ev.seeds))
        if selection and selection[role]>=0 and roles[role] is not None:
            init=roles[role].state[:scene.nseed,selection[role]]
        w=(ev.affinity[role],top16(ev.affinity[role]),r88,r91_evidence(ev,role,old_inside),init,
           p['lcib_support'][role])
        answer.append({stage:support_owner(weights,known,n_obj) for stage,weights in zip(STAGES,w)})
    return answer


def parser():
    sys.path.insert(0,str(ROOT/'research/round97'))
    import torch
    import parse97
    path=ROOT/'crossplatform/round20/assets'
    keywords=read(path/'STANZA_ASSET_RECEIPT.json')['kwargs']
    torch.set_num_threads(2)
    nlp=parse97.stanza.Pipeline('en',dir=str(path/'stanza'),processors='tokenize,mwt,pos,lemma,depparse',
            package=None,use_gpu=False,download_method=None,verbose=False,**keywords)

    def graph(text):
        doc=parse97.dependency_document(nlp(text))
        subject=parse97.parse_subject(text,doc)
        return parse97.repair_inventory87(text,doc,parse97.parse85(text,doc,subject)),doc
    return graph


def parsed_edges(ev,text,graph,doc):
    names=[e['role'] for e in graph['entities']]
    assert [list(e['span']) for e in graph['entities']]==ev.role_spans.tolist()
    retained=[]
    for edge in graph['relations']:
        if edge['known_frame'] and edge['predicate'] in ('LEFT','RIGHT'):
            a,b=names.index(edge['subject']),names.index(edge['reference'])
            direction=1 if edge['predicate']=='LEFT' else -1
            w=next(w for w in doc['sentences'][edge['sentence']]['words'] if w['id']==edge['governor_word'])
            st,en=w['start_char'],w['end_char']
            context=text[max(0,st-33):min(len(text),en+45)]
            retained.append((a,b,direction,context))
    assert [list(r[:3]) for r in retained]==ev.edges.tolist()
    return retained


def extract():
    registered=read(PARENT_WORK/'AUDIT_PANEL_PRIVATE.json')
    assert sha(PARENT_WORK/'AUDIT_PANEL_PRIVATE.json')==read(PARENT/'PANEL_LOCK.json')['input_sha256']
    old_seal=read(PARENT/'DEV_SEAL.json')
    assert sha(PARENT_WORK/'DEV_FILES_PRIVATE.json')==old_seal['file_index_sha256']
    files=read(PARENT_WORK/'DEV_FILES_PRIVATE.json')
    rows=target_rows()
    specs=read(SETTINGS)['sources']
    pairs=Counter()
    selected=[]
    rerun_parser=None
    for i in registered:
        item=rows[i]
        row=item['r']
        target=item['target']
        source=item['source']
        name=f'{i:05d}.npz'
        path=PARENT_WORK/'DEV'/name
        assert sha(path)==files[name]
        with np.load(path,allow_pickle=False) as p:
            ev=read_evidence(p)
            if not len(ev.edges):
                continue
            names,boxes=annotation_inventory(row['metapath'],target,int(row['suffix'] or 0),int(row['object_index']))
            owners=seed_owners(ev.seeds,boxes)
            old=np.r_[ev.boxes[ev.old_winner],0.]
            inside=in_box(ev.seeds,old)
            ids=[support_owner(r91_evidence(ev,k,inside),owners,len(boxes)) for k in range(len(ev.affinity))]
            fitted=[support_owner(w,owners,len(boxes)) for w in p['lcib_support']]
            collisions=[]
            for j,(a,b,s) in enumerate(ev.edges.tolist()):
                for name,support in [('R91',ids),('LCIB',fitted)]:
                    pairs[name,'total']+=1
                    if min(support[a],support[b])>=0:
                        pairs[name,'known']+=1
                        pairs[name,'merged']+=int(support[a]==support[b])
                if ids[a]>=0 and ids[a]==ids[b]:
                    collisions.append(j)
            if not collisions:
                continue
            if rerun_parser is None:
                rerun_parser=parser()
            text=sensor_row(row)['caption']
            graph,doc=rerun_parser(text)
            relations=parsed_edges(ev,text,graph,doc)
            stages=role_stages(ev,p,specs[source],owners,len(boxes),old)
            pixel=CameraX(ev.camera_intrinsic,ev.model_to_camera,ev.distortion,ev.width)
            scene_boxes=[]
            for name,b in zip(names,boxes):
                u=pixel.project(b[:3])[0]
                scene_boxes.append({'class':name,'u':float(u) if np.isfinite(u) else None})
            for j in collisions:
                a,b,d,context=relations[j]
                phrases=[text[s:t] for s,t in ev.role_spans.tolist()]
                auto_labels=[0 if k==0 else find_expected_ref(ev,names,boxes,text,target,k) for k in (a,b)]
                entry=dict(case=len(selected)+1,dev_ordinal=item['ordinal'],panel_index=i,relation_index=j,
                           source=source,target=target,phrase=text,role_phrases=[phrases[a],phrases[b]],role_indices=[a,b],
                           parsed='LEFT' if d>0 else 'RIGHT',cue=context,boxes=scene_boxes,
                           automatic_candidates=auto_labels,shared_r91=ids[a],
                           stages={stage:[stages[a][stage],stages[b][stage]] for stage in STAGES})
                assert entry['stages']['r91_init']==[ids[a],ids[b]]
                selected.append(entry)
    prior=table(PARENT/'results/DEV/role_overlap_proxy.csv')
    assert [(pairs[r['readout'].upper(),x],int(r[k])) for r in prior for x,k in (
        ('total','pairs'),('known','known'),('merged','merged'))]==[(132,132),(38,38),(33,33),(132,132),(25,25),(21,21)]
    assert len(selected)==33
    path=WORK/'ROLE_REVIEW_PRIVATE.json'
    atomic_json(path, selected)
    physical={rows[r['panel_index']]['point_sha'] for r in selected}
    atomic_json(HERE/'ROLE_SCOPE.json',dict(rows=33,directed_expressions=len({r['panel_index'] for r in selected}),
                    physical_point_observations=len(physical), original_R91=dict(total=132,known=38,merged=33),
                    original_LCIB=dict(total=132,known=25,merged=21),
                    matched_parent_native_parser=True,private_review_sha256=sha(path),FINAL_access=False))
    print('Reproduced the 33 shared edges in',len(physical),'physically distinct point clouds',flush=True)


def render():
    records=read(WORK/'ROLE_REVIEW_PRIVATE.json')
    rows=target_rows()
    directory=WORK/'role_review_PRIVATE'
    directory.mkdir(exist_ok=True)
    font=ImageFont.truetype('DejaVuSans.ttf',16)
    small=ImageFont.truetype('DejaVuSans.ttf',12)
    excerpts=[]
    cards=[]
    for record in records:
        i=record['panel_index']
        row=rows[i]['r']
        names,boxes=annotation_inventory(row['metapath'],record['target'],int(row['suffix'] or 0),int(row['object_index']))
        with np.load(PARENT_WORK/'DEV'/f'{i:05d}.npz',allow_pickle=False) as f:
            ev=read_evidence(f)
        im=Image.open(row['imagepath']).convert('RGB')
        im.thumbnail((674,370))
        w,h=im.size
        draw=ImageDraw.Draw(im)
        for pos,box in enumerate(boxes):
            col='#51ef88' if pos==0 else '#ffad54' if pos==record['shared_r91'] else '#62c5ff'
            uv=projection(ev,corners(box))*[w,h]
            for a,b in LINES:
                if np.isfinite(uv[[a,b]]).all():
                    draw.line([tuple(uv[a]),tuple(uv[b])],fill=col,width=2)
            u=projection(ev,box[None,:3])*[w,h]
            x,y=u[0]
            if np.isfinite(x+y) and 0<=x<w and 0<=y<h:
                y=max(12.,y-14)
                draw.text((x,y),str(pos),fill=col,font=font,stroke_width=2,stroke_fill='#000000')
        card=Image.new('RGB',(694,438),'white')
        head=ImageDraw.Draw(card)
        title=f'{record["case"]:02d}  {record["source"]} -> {record["target"]}  |  target #0  |  shared R91 #{record["shared_r91"]}'
        head.text((10,7),title,font=font,fill='#162227')
        card.paste(im,((694-w)//2,32))
        summary=f'{record["role_phrases"][0]}   {record["parsed"]} of   {record["role_phrases"][1]}'
        head.text((10,408),summary[:102],font=small,fill='#18242d')
        cards.append(card)
        locality='; '.join(f'{j}={x["class"]}@{x["u"]:.2f}' if x['u'] is not None else f'{j}={x["class"]}@off'
                           for j,x in enumerate(record['boxes']))
        excerpts.append(f'{title}\n'
                        f'roles {record["role_indices"]}: {summary}\n'
                        f'cue: {record["cue"]}\n'
                        f'{record["phrase"]}\n'
                        f'labels: {locality}\n'
                        f'auto expected: {record["automatic_candidates"]}, stages: {record["stages"]}\n')
    for j in range(0,len(cards),4):
        group=cards[j:j+4]
        page=Image.new('RGB',(1388,438*((len(group)+1)//2)),'white')
        for k,card in enumerate(group):
            page.paste(card,((k%2)*694,(k//2)*438))
        page.save(directory/f'{j+1:02d}-{j+len(group):02d}.jpg',quality=92)
    (directory/'READING_PRIVATE.txt').write_text('\n'.join(excerpts))
    print('All 33 local-only visual cards rendered',flush=True)


if __name__=='__main__':
    extract()
    render()
