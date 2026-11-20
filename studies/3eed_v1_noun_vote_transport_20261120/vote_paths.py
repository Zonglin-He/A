"""Immutable input bindings. The catalog reads observations and past label-free decisions only."""
import hashlib
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'3eed_v1_physical_tree_role_decoding_20261120'
sys.path.insert(0,str(PREVIOUS))
import paths_tree as old
from paths_tree import ROOT, SOURCES, CLASSES, read, save, digest, archive_npz
from full_roster import FULL
from full_observe import key
from panel_nouns import EXTRA
from predict_nouns import inputs as noun_inputs
from scope183 import ordinary,frame

SPEC=HERE/'PROTOCOL.json'
CFG=read(SPEC)
ARMS=tuple(CFG['arms'])
WORK=HERE/'private'
OUT=HERE/'results'
CATS=len(CLASSES)-1

def pins():
    files={'frozen_registry':ROOT/'crossplatform/CURRENT_METHOD.json','source_registry':ROOT/'crossplatform/single_frame_sources.json',
           'small_seal':old.PRED/'SEALED.json','ordinary_seal':FULL/'predictions/SEALED.json','single_word_seal':EXTRA/'SEALED.json'}
    result={k:digest(f) for k,f in files.items()}
    assert all(CFG[k]==v for k,v in result.items())
    return result

def frame_id(src,f):
    return hashlib.sha256(repr(('noun-delivery/1',src,key(f))).encode()).hexdigest()[:28]

def make():
    base=pins()
    frames={};rows=[]
    def add(scope,src,uid,f,words,role,rh,lex,lh,word,wh,prior,ph,archive_frame=None,ix=-1):
        ident=frame_id(src,f)
        if word is not None:
            if ident not in frames:
                frames[ident]=dict(f=f,source=src,lex=str(lex),lh=lh,word=str(word),wh=wh,tree=archive_frame)
            elif archive_frame:
                frames[ident]['tree']=archive_frame
        rows.append(dict(scope=scope,source=src,target=f['target'],uid=uid,frame=ident,ordinal=int(ix),
                         words=list(map(int,words)),role=str(role),rh=rh,prior=str(prior),ph=ph,active=word is not None))
    prior=read(old.PRED/'SEALED.json')['files']
    obj=read(old.OBJECTS/'SEALED.json')['files']
    wordseal=read(old.parent.OBS/'SEALED.json')['hashes']
    for item in old.observations():
        row,rp,rh,lp,lh,fid,words,*_=item
        name=old.name(row); f=row['frame']
        assert digest(rp)==rh
        with np.load(rp) as z:active=bool(z['active']) and min(words)>=0
        wp=old.parent.OBS/(fid+'.npz') if active else None
        add('panel',row['source'],row['uid'],f,words,rp,rh,lp,lh,wp,wordseal[fid] if active else None,
            old.PRED/name,prior[name],[str(old.OBJECTS/(fid+'.npz')),obj[fid]] if active else None)
    seals={s:read(EXTRA/s/'SEALED.json')['files'] for s in SOURCES}
    for row in noun_inputs():
        src=row['source'];f=row['frame'];name=old.name(row);entry=seals[src][name];fid=entry['fid'];folder=EXTRA/src
        from observe_witness import interpret
        roles=interpret(f['text'],{})
        words=[CLASSES.index(v) for v in roles[:2]]
        add('noun',src,row['uid'],f,words,folder/(name[:-4]+'_native.npz'),entry['native'],
            folder/(fid+'_lex.npz'),entry['lex'],folder/(fid+'_word.npz'),entry['word'],folder/name,entry['pred'])
    parsed=read(FULL/'parsed.json'); tab=ordinary()
    whole=read(FULL/'predictions/SEALED.json')
    for src in SOURCES:
        rec=read(FULL/'observations'/(src+'.json'))
        for tgt in SOURCES:
            if src==tgt:continue
            direction=src+'_to_'+tgt
            folder=FULL/'predictions'/direction
            assert digest(folder/'SEALED.json')==whole['dirs'][direction]
            files=read(folder/'SEALED.json')['files']
            for i,role in enumerate(parsed[tgt]):
                if role is None:continue
                v=rec['records'][tgt+'_'+str(i)];features=rec['frames'][v['frame']]
                add('ordinary',src,direction+'_'+str(i),frame(tab[tgt][i]),role[:2],v['role'],v['sha'],features['lex'],
                    features['lex_sha'],features['word'],features['word_sha'],folder/(str(i)+'.npz'),files[str(i)],ix=i)
    assert len({(r['scope'],r['source'],r['uid']) for r in rows})==len(rows)
    assert tuple(sum(r['scope']==k for r in rows) for k in ('panel','noun','ordinary'))==(1204,128,1294)
    save(WORK/'catalog.json',dict(frames=frames,rows=rows))
    save(WORK/'CATALOG.json',dict(inputs=base,code=digest(Path(__file__)),protocol=digest(SPEC),catalog=digest(WORK/'catalog.json'),
         rows=len(rows),frame_sources=len(frames),labels_used=False))
    print('records',len(rows),'unique active source frames',len(frames),flush=True)

def catalog():
    checks=read(WORK/'CATALOG.json')
    assert checks['protocol']==digest(SPEC) and checks['inputs']==pins() and checks['code']==digest(Path(__file__))
    assert checks['catalog']==digest(WORK/'catalog.json')
    return read(WORK/'catalog.json')

def row_id(row):
    return hashlib.sha256(repr((row['scope'],row['source'],row['uid'])).encode()).hexdigest()[:28]

if __name__=='__main__':
    make()

