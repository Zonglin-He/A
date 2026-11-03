"""Independently reconstruct every subset without importing its producer."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);out=ROOT/p['output'];seal=read(out/'SEALED.json');assert seal['plan_sha256']==sha(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 for rel,h in seal['files'].items():assert sha(out/rel)==h
 rows=read(ROOT/p['manifest']);arms=p['arms'];statuses=read(out/'READOUT_STATUS_PRIVATE.json');assert [r['id'] for r in rows]==[r['id'] for r in statuses]
 checks=0
 for row,st in zip(rows,statuses):
  sid=row['id'];zs={}
  for k in ['new_cache','old_cache']:
   folder=ROOT/p[k];ss=read(folder/'SEALED.json');rel='samples_PRIVATE/'+sid+'.npz';assert sha(folder/rel)==ss['files'][rel];zs[k]=np.load(folder/rel,allow_pickle=False)
  new,old=zs['new_cache'],zs['old_cache'];source=ROOT/'crossplatform/round24/capture336_attempt001';cs=read(source/'SEALED.json');rel='samples_PRIVATE/'+sid+'.npz';assert sha(source/rel)==cs['files'][rel]
  with np.load(source/rel,allow_pickle=False) as canon,np.load(out/rel,allow_pickle=False) as pred:
   same(canon['boxes'],pred['source_boxes']);same(canon['scores'],pred['source_scores']);same(new['source_boxes'],old['source_boxes'])
   s=canon['scores'];oi=old['hybrid_arm_names'].tolist().index('replace_appearance_c05');cubes={'new':new['components'][0],'old':old['hybrid_components'][oi]};known={'new':new['cue_query_known'][0],'old':old['hybrid_cue_query_known'][oi]}
   assert pred['arm_names'].tolist()==arms and set(st['arm_statuses'])==set(arms)
   for j,arm in enumerate(arms):
    active=False
    if arm=='source':expected=s.astype('float64')
    elif arm=='external_G':expected=old['scores'][old['arm_names'].tolist().index(arm)];active=True
    else:
     fam,subset=arm.split('_');v=cubes[fam];expected=s.astype('float64')
     if 'V' in subset:expected=expected+v[0]*2;active|=bool(known[fam][0].any())
     if 'P' in subset:expected=expected+v[1]*2;active|=bool(known[fam][1].any())
     if 'R' in subset:expected=expected+v[2]*2;active|=bool(known[fam][2].any())
    same(expected,pred['scores'][j]);assert pred['queries'][j]==int(np.argmax(expected));same(canon['boxes'][int(np.argmax(expected))],pred['selected_original_boxes'][j])
    unavailable=not active if arm not in ['source','external_G'] else False
    assert st['arm_statuses'][arm]=={'unknown':unavailable,'fallback':unavailable,'no_parameter_update':True};checks+=1
  for z in zs.values():z.close()
 result={'state':'PASS_R52_INDEPENDENT_CACHE_CUE_ARITHMETIC_AND_CANONICAL_READOUT','n':len(rows),'arm_names':arms,'prediction_seal_sha256':sha(out/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'candidate_score_vectors_checked':checks,'GT_reads':0,'raw_media_reads':0,'model_forwards':0}
 dest=ROOT/'research/round52/prediction_audit_attempt001';assert not dest.exists();dest.mkdir();(dest/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
