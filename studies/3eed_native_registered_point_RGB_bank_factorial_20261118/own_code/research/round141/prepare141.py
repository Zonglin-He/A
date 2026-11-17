"""Only ordinary inherited sensors/registrations, no new media/model or GT access."""
from common141 import *
import numpy as np,collections
def main():
 check();reg=rd(R/'research/round135/INPUT_REGISTRY_PRIVATE.json');anchors={a['A']['frame_PRIVATE']:a for a in rd(R/'research/round131/ANCHORS_PRIVATE.json')};pairs={a['pair_PRIVATE']:a for a in rd(R/'research/round131/PAIRS_PRIVATE.json')};out=[];eng=[];counts=collections.Counter()
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;an=anchors[e['frame_PRIVATE']];A=an['A'];assert A['eligible_for_extra_media'] and not A['protected_image_content_alias'];D=R/'crossplatform/round135/formal'/name;f=np.load(D/'features_PRIVATE'/(sid+'.npz'));has=bool(f['valid_B'].any());rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert has==rec['has_B'];ee=dict(e,A_sensor=A,has_B=has)
  if has:
   pair=pairs[an['B_pair']];sensor=pair['B'];assert sensor['eligible_for_extra_media'] and not sensor['protected_image_content_alias'];rp='crossplatform/round131/registration_PRIVATE/'+an['B_pair']+'.json';rr=rd(R/rp);assert rr['geometry_cycle_supported'] and rr['A_sensor']['point_sha256']==A['point_sha256'] and rr['B_sensor']['point_sha256']==sensor['point_sha256'];ee.update(pair_PRIVATE=pair,registration_path=rp,registration_sha256=sha(R/rp))
  out.append(ee);counts['n']+=1;counts['B_available']+=has;counts['no_B_exact_A']+=not has
  if not any(x['source']==src and x['target']==tgt and x['has_B']==has for x in eng):eng.append(ee)
 assert len(out)==1798 and counts['B_available']==764 and len(eng)==12;dump(H/'INPUT_REGISTRY_PRIVATE.json',out);dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(H/'PREPARATION.json',dict(counts)|{'GT_or_media_read':False,'source_training':False,'FINAL_access':False});dump(H/'experiment/INPUT_LOCK.json',{'files':{str(p.relative_to(R)):sha(p) for p in [H/'INPUT_REGISTRY_PRIVATE.json',H/'ENGINEERING_INPUTS_PRIVATE.json']},'selector':'first B/noB per source-target direction, no scores/GT'});print(dict(counts),flush=True)
if __name__=='__main__':main()
