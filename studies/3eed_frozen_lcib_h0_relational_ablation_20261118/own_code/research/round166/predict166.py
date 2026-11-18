"""Replay frozen LCIB input evidence on R165 and seal the five h0 readouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gc
import hashlib
import multiprocessing
import time

import numpy as np

from scope166 import (ARMS, BLOCK, HERE, ORDER, PLATFORMS, PROTOCOL, SETTINGS, W97, W165,
                      WORK, atomic_json, inputs, lock, npz_atomic, read, sha, status)
from evidence162 import Evidence, EvidenceGrounder, attention_weights
from init166 import frozen_case
from predict165 import cached, points_for
from common160 import array_sha


@lru_cache(maxsize=512)
def camera(rowpath, imagepath, target, view):
    from PIL import Image
    from camera_projection import CameraSpec, model_to_camera_matrix, read_camera_allowlist
    with Image.open(imagepath) as im:
        width,height=im.size
    spec=CameraSpec.from_camera_fields(target,read_camera_allowlist(rowpath),waymo_view_suffix=view,
                                      missing_distortion_policy='assume_zero_declared')
    intrinsic,_,distortion,_=spec.validated()
    return intrinsic,model_to_camera_matrix(spec),distortion,width,height


def capture_case(model, row, baseline, raw, points_hash, index_hash, verify):
    import torch
    from contextlib import ExitStack
    from grounding158 import select_grounding_box
    from interface66 import capture
    from model76 import tensor_data, tensor_prefix
    from routing88 import build88, token_roles88
    from crossplatform.ground_single_frame import orient_box

    start=time.perf_counter()
    text=row['caption']
    parsed=read(W97/'roles_PRIVATE'/f"{row['id']}.json")
    assert parsed['caption_sha256']==hashlib.sha256(text.encode()).hexdigest()
    roles,doc=parsed['graph'],parsed['doc']
    points,indices=points_for(row)
    assert array_sha(points)==points_hash.tobytes().hex() and array_sha(indices)==index_hash.tobytes().hex()
    seeds_order=[]

    def seed_order(_module,_args,result):
        first,second=result['sa_ins'][1:3]
        seeds_order.append(first.long().gather(1,second.long())[0].cpu().numpy())

    with model._lock,torch.inference_mode():
        with ExitStack() as hooks:
            hooks.callback(model.base.backbone_net.register_forward_hook(seed_order).remove)
            z,proof=capture(model.base,points,indices,text)
        np.testing.assert_array_equal(z['boxes'],raw)
        seeds=z['xyz'][0]
        seed_index,=seeds_order
        np.testing.assert_array_equal(seeds,points[seed_index])
        spatial,_,probability=cached(model.source,row,z,proof,verify)
        prefix,data=tensor_prefix(z),tensor_data(spatial,row)
        role_mask,_=token_roles88(model.base,row,z,roles,doc)
        observed={}

        def record(name, fn):
            def hook(module,args,out):
                observed[name]=fn(module,args,out).cpu().numpy()
            return hook

        def local(module,args,out):
            return attention_weights(module,args,out,verify)

        matcher=model.matcher
        assert matcher.route is None and not matcher.use_interface
        layer=model.base.cross_encoder.layers[-1].cross_layer.cross_lv
        with ExitStack() as hooks:
            hooks.callback(layer.register_forward_hook(record('attention',lambda _m,_a,x:x[1][0])).remove)
            hooks.callback(matcher.image_to_point.register_forward_hook(record('p2i',local)).remove)
            hooks.callback(matcher.point_to_image.register_forward_hook(record('i2p',local)).remove)
            hooks.callback(matcher.image_adapter.register_forward_hook(record('ionly',lambda _m,_a,x:x)).remove)
            initial,_,_=matcher.main(model.base,prefix,data,'bidirectional')
        attention=observed['attention']
        route,_=build88(row,dict(z,frozen_probability=initial.cpu().numpy()),spatial,roles,role_mask,attention)
        ids={e['role']:i for i,e in enumerate(roles['entities'])}
        edges=[(ids[e['subject']],ids[e['reference']],1 if e['predicate']=='LEFT' else -1)
               for e in roles['relations'] if e['known_frame'] and e['predicate'] in ('LEFT','RIGHT')]
        offsets=np.array([e['span'] for e in roles['entities']],np.int16).reshape(-1,2)
        winner,utility=select_grounding_box(raw,probability)
        np.testing.assert_array_equal(orient_box(points,raw[winner]),baseline)
        K,T,D,W,H=camera(row['metapath'],row['imagepath'],row['platform'],int(row['suffix'] or 0))
        maps={n:spatial[n] for n in ('p2i','p2i_weight','i2p','i2p_weight')}
        affinity=(role_mask@attention)/np.maximum(1,role_mask.sum(1))[:,None]
        ev=Evidence(boxes=raw,probability=probability,utility=utility,old_winner=winner,old_pca=baseline,
                    points=points,indices=indices,seeds=seeds,seed_sample_indices=seed_index,
                    anchors=z['query_points_sample_inds'],role_mask=role_mask,role_spans=offsets,
                    edges=np.array(edges,np.int16).reshape(-1,3),affinity=affinity.astype(np.float32),
                    supports=route['supports'],hypotheses=route['hypotheses_PRIVATE'],route_weights=route['weights_plain'],
                    subject_weights=np.empty((0,0),np.float32),p2i_learned=observed['p2i'],i2p_learned=observed['i2p'],
                    ionly=observed['ionly'],visible=spatial['visible'],uv=spatial['uv_grid']/[48,32],
                    depth=spatial['point_depth'],camera_intrinsic=K,model_to_camera=T,distortion=D,width=W,height=H,
                    forward_seconds=time.perf_counter()-start,**maps)
    if verify:
        original=model.evidence_frame(row,verify=True)
        for name in ('boxes','points','indices','seeds','seed_sample_indices','role_mask','role_spans','edges','affinity',
                     'supports','hypotheses','route_weights','p2i_learned','i2p_learned','ionly','visible','old_pca','uv','depth'):
            np.testing.assert_array_equal(getattr(ev,name),getattr(original,name),err_msg=name)
        np.testing.assert_allclose(ev.probability,original.probability,rtol=0,atol=1e-10)
    return ev,np.mean(attention,axis=0,dtype=np.float32)


def freeze(ev, shared, config, verify):
    record=frozen_case(ev,shared,config,verify)
    record['forward_seconds']=ev.forward_seconds
    return record


def save_block(path,predictions,ordinals,parent_sha):
    data={k:np.stack([p[k] for p in predictions]) for k in predictions[0] if k!='edges'}
    edges=[p['edges'] for p in predictions]
    npz_atomic(path,**data,edges=np.concatenate(edges),edge_offsets=np.r_[0,np.cumsum([len(e) for e in edges])],
               ordinals=ordinals,arms=np.array(ARMS),parent_block_sha256=parent_sha,protocol=sha(PROTOCOL))


def run(pilot):
    lock()
    rows,parent=inputs()
    config=read(SETTINGS)['sources']
    if not pilot:
        probe=read(HERE/'PILOT_SEAL.json')
        assert probe['n']==48 and probe['native_h0_and_frozen_evidence_replayed']
        status('running',stage='frozen_prediction',protocol='round166/experiment/PROTOCOL.json')
    from model51 import fingerprint
    import torch
    cohort='pilot' if pilot else 'ordinary'
    start=time.perf_counter()
    saved={}
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for source in PLATFORMS:
            model=EvidenceGrounder(source)
            digest=(fingerprint(model.base),fingerprint(model.matcher))
            for s,target in ORDER:
                if s!=source:
                    continue
                stem=f'{s}_to_{target}'
                predecessor=W165/'ordinary'/stem
                ref=read(predecessor/'SEAL.json')
                dest=WORK/cohort/stem
                dest.mkdir(parents=True,exist_ok=True)
                pca=parent['per_direction'][stem]
                assert sha(predecessor/'SEAL.json')==pca['direction_forward_seal']
                keep=set(np.linspace(0,len(rows[target])-1,8,dtype=int).tolist()) if pilot else None
                total=0
                files={}
                for key,known in sorted(ref['files'].items()):
                    prior=predecessor/key
                    assert sha(prior)==known
                    with np.load(prior,allow_pickle=False) as block:
                        ordinals=block['ordinals'].tolist()
                        mask=[k for k,i in enumerate(ordinals) if keep is None or i in keep]
                        if not mask:
                            continue
                        ix=[ordinals[k] for k in mask]
                        raw,boxes,ps,ih=(block[k] for k in ('raw','boxes','points_sha','indices_sha'))
                    out=dest/key
                    if not out.exists():
                        pending=[]
                        for k,i in zip(mask,ix):
                            ev,shared=capture_case(model,rows[target][i],boxes[k,-1],raw[k],ps[k],ih[k],pilot)
                            pending.append(pool.submit(freeze,ev,shared,config[source],pilot))
                        data=[item.result() for item in pending]
                        save_block(out,data,ix,known)
                    else:
                        with np.load(out,allow_pickle=False) as data:
                            assert data['ordinals'].tolist()==ix and str(data['parent_block_sha256'])==known
                            assert str(data['protocol'])==sha(PROTOCOL)
                    files[key]=sha(out)
                    total+=len(ix)
                    print(cohort,stem,total,'/',8 if pilot else pca['n'],'sec',round(time.perf_counter()-start,1),flush=True)
                assert total==(8 if pilot else pca['n'])
                atomic_json(dest/'SEAL.json',dict(n=total,files=files,protocol_sha256=sha(PROTOCOL),
                            target_label_read=False,FINAL_access=False,parent_forward_sha256=sha(predecessor/'SEAL.json')))
                saved[stem]=dict(n=total,seal=sha(dest/'SEAL.json'))
            assert digest==(fingerprint(model.base),fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(HERE/('PILOT_SEAL.json' if pilot else 'PREDICTION_SEAL.json'),dict(n=sum(v['n'] for v in saved.values()),
                directions=saved,protocol_sha256=sha(PROTOCOL),FINAL_access=False,target_label_read=False,
                native_h0_and_frozen_evidence_replayed=pilot,parent_zero_step_unchanged=True,
                no_support_sweeps_or_geometry_fits=True,seconds=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    run(p.parse_args().pilot)
