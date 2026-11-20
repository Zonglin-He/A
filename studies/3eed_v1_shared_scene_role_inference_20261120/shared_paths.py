"""Workspace and immutable observed-frame inputs for the joint physical-scene study."""
from pathlib import Path
from hashlib import sha256
import sys

HERE=Path(__file__).resolve().parent
LEX=HERE.parent/'3eed_v1_physical_noun_evidence_20261120'
sys.path.insert(0,str(LEX))
from common_lex import ROOT, LAST, OLD, SOURCES, CLASSES, name, read, save, digest, archive_npz, panel

WORK=HERE/'private'
OUT=HERE/'results'
SPEC=HERE/'PROTOCOL.json'
CFG=read(SPEC)
NEW=WORK/'additional'
SEG=WORK/'components'
PRED=WORK/'predictions'
ARMS=tuple(CFG['arms'])

def anonymous(x):
    return sha256(('shared-role-scene/1|'+str(x)).encode()).hexdigest()[:22]

def observed_roster():
    base=panel()
    assert digest(LEX/'private/observations/SEALED.json')==CFG['old_semantics']
    assert digest(LEX/'private/predictions/SEALED.json')==CFG['old_predictions']
    assert digest(OLD/'observed/SEALED.json')==CFG['old_role_observations']
    assert digest(OLD/'roster.json')==CFG['old_roster']
    cert=read(OUT/'panel.json')
    assert digest(NEW/'roster.json')==cert['observations']
    rows=base+read(NEW/'roster.json')
    assert len({name(r) for r in rows})==len(rows)
    return rows

def inputs():
    rows=observed_roster()
    os=read(LEX/'private/observations/SEALED.json')
    obs=read(OLD/'observed/SEALED.json')['files']
    pp=read(LEX/'private/predictions/SEALED.json')['files']
    new=read(NEW/'SEALED.json')
    assert new['roster']==read(OUT/'panel.json')['observations']
    for row in rows:
        n=name(row)
        if n in os['files']:
            fid=os['files'][n]; base=LEX/'private/observations'/(fid+'.npz')
            yield row,(OLD/'observed'/n),obs[n],base,os['hashes'][fid],fid,os['nouns'][n],LEX/'private/predictions'/n,pp[n]
        else:
            v=new['files'][n];fid=v['fid']
            yield row,NEW/'role'/n,v['sha'],NEW/'lexical'/(fid+'.npz'),new['hashes'][fid],fid,v['words'],None,None
