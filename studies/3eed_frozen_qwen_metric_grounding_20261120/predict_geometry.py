import hashlib
import json
from pathlib import Path
import numpy as np
from paths import PRIV,OUT,SPEC,read,write,digest
from geometry import ARMS,camera,raw_returns,variants,to_sensor,intersections

def run():
    hashes=read(PRIV/'reply_hashes.json')
    seal=read(OUT/'model_seal.json')
    assert seal['pins']['protocol']==digest(SPEC) and seal['ordered_reply_hash']==hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    assert seal['pins']['panel']==digest(PRIV/'panel.json')
    result={}
    for row in read(PRIV/'panel.json'):
        uid=row['uid'];vals={}
        for mode in ('2d','3d'):
            k=uid+'_'+mode;p=PRIV/'replies'/(k+'.json')
            assert digest(p)==hashes[k]
            reply=read(p);vals[mode]=reply['value']
        spec,size,mat=camera(row)
        d=dict(pixel=vals['2d'],camera_boxes={},sensor_boxes={},surface={},support={'fit_bins':0,'heldout_bins':0,'wrong_fit_bins':0})
        assert digest(row['points'])==row['point_sha']
        if vals['3d'] is not None:
            uv,points=raw_returns(row,spec,size)
            boxes,d['support'],held=variants(np.array(vals['3d']),vals['2d'],spec,size,uv,points)
            for arm,b in boxes.items():
                d['camera_boxes'][arm]=b.tolist()
                d['sensor_boxes'][arm]=to_sensor(b,mat)[0].tolist()
                hits,errors=intersections(b,held)
                d['surface'][arm]=[float(e) if h else None for h,e in zip(hits,errors)]
        result[uid]=d
    write(PRIV/'predictions.json',result)
    write(OUT/'geometry_seal.json',dict(predictions=digest(PRIV/'predictions.json'),model=digest(OUT/'model_seal.json'),
        geometry=digest(Path(__file__).with_name('geometry.py')),code=digest(__file__),n=len(result),GT_read=False))

if __name__=='__main__':run()
