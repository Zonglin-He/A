"""Seal every hypothesis and answer before joining the separate evaluation key."""
from pathlib import Path
import numpy as np
from io_scene import HERE, SPEC, OUT, PRIVATE, PLATFORMS, read, write, digest, key
import belief_scene
import spatial_simulator as sim

def run():
    rows=read(PRIVATE/'questions.json')
    panel=read(OUT/'panel_seal.json')
    assert digest(PRIVATE/'questions.json')==panel['questions'] and digest(SPEC)==panel['protocol']
    code={f.name:digest(f) for f in [Path(__file__),Path(sim.__file__),Path(belief_scene.__file__),HERE/'observe_scene.py',HERE/'io_scene.py']}
    files={};inputs={};draws=0;cardinalities=[]
    for source in PLATFORMS:
        obs=PRIVATE/'observations'/source
        source_seal=read(obs/'seal.json')
        assert source_seal['pins']['questions']==panel['questions'] and source_seal['pins']['observer']==code['observe_scene.py']
        inputs[source]=digest(obs/'seal.json')
        pred=PRIVATE/'predictions'/source
        pred.mkdir(parents=True,exist_ok=True)
        for row in rows:
            if row['frame']['target']==source:continue
            uid=row['uid']
            op=obs/(uid+'.npz'); assert digest(op)==source_seal['files'][uid]
            with np.load(op) as d:
                indices,compatible,dist,confidence=belief_scene.build(d,row['higher'],int(key('perm',source,uid),16))
                events=sim.outcomes(d['subject_boxes'][indices],d['reference_boxes'][indices],d['camera'])
                answers=np.stack([sim.pushforward(p,events) for p in dist])
                conf=np.abs(2*answers-1)
                for i,c in enumerate(confidence):conf[i]=c
                # Structural diagnostics cover the full forest, even where there is no legal pair.
                blocked=[float(p[~compatible].sum()) for p in dist]
                total=np.array([p.sum() for p in dist]);assert np.all(np.isclose(total,1)|np.isclose(total,0))
                rng=np.random.default_rng(int(key('world_audit',source,uid),16))
                examples=[]
                for _ in range(8):
                    world=belief_scene.sample_world(d['parent'],d['stop'],indices,dist[5],rng)
                    if world is None:break
                    occupied,pair=world
                    active=set(occupied)
                    for node in active:
                        a=d['parent'][node]
                        while a>=0:
                            assert a not in active
                            a=d['parent'][a]
                    assert set(pair)<=active
                    examples.append([len(active),*pair]);cardinalities.append(len(active));draws+=1
                path=pred/(uid+'.npz');tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,p=dist,answers=answers,confidence=conf,invalid=blocked,
                     ids=indices,no_mass=(total==0),visible_count=d['visible_count'][indices],
                     nodes=len(d['parent']),sampled_cut_diagnostics=np.asarray(examples,dtype=int).reshape(-1,3),size=d['size'][indices],sb=d['subject_boxes'][indices],
                     rb=d['reference_boxes'][indices],camera=d['camera'])
                tmp.replace(path)
            files[source+'/'+uid]=digest(path)
        print('sealed predictions',source,flush=True)
    assert len(files)==2*len(rows)
    write(OUT/'prediction_seal.json',dict(protocol=digest(SPEC),panel=digest(OUT/'panel_seal.json'),code=code,
          source_captures=inputs,files=files,source_frames=len(files),physical_frames=len(rows),
          conditional_full_cuts_checked=draws,median_occupied_nodes=float(np.median(cardinalities)) if cardinalities else None,model_weights_updated=False,answer_key_read=False,FINAL_access=False))
if __name__=='__main__':run()
