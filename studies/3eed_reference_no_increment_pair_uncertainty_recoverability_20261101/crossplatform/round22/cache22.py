"""Remove modifier-vs-head veto with all other evidence fixed; no fresh inference."""
import copy,numpy as np
from common22 import *
from observations import load
from assemble21_v3 import relation_evidence,decision
ARMS=['source','r21_no_ref','r21_full','drop_increment_no_ref','drop_increment_full','conservative_no_ref','conservative_full']
def refs_without_increment(r):
    x=copy.deepcopy(prior(r,'reference'))
    for obs in x['references'].values():
        for o in obs:o['semantic_accepted']=bool(o['full']['log_odds_yes_no']>=.5 and o['visual_evidence']>=.5)
    return x
def main():
    lock('CACHE_LOCK.json');out=BASE/'cache208_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();(out/'predictions_PRIVATE').mkdir()
    for r in read(ROWS):
        z=feature(r);q0=int(z['scores'].argmax());sel={a:q0 for a in ARMS};g=prior(r,'text');d=prior(r,'dino');ref=refs_without_increment(r);case=prior(r,'direct');prep=old(r,'pairs');diag=[]
        with np.load(PRIOR/'direct208_attempt003/predictions_PRIVATE'/(r['id']+'.npz')) as zz:
            names=zz['arm_names'].tolist();sel['r21_no_ref']=int(zz['query_indices'][names.index('gate_repair_no_reference')]);sel['r21_full']=int(zz['query_indices'][names.index('direct_full_relation_G')])
        if prep['pairs']:
            frame,scene,binding=load(r)
            for pair,p in zip(prep['pairs'],case['pair_diagnostics']):
                rr=relation_evidence(g,scene,d,ref,pair);delta=p['full_delta'];flags={};strength={};visual=p['visual_gate_repaired'];weak=p['weak_visual_compatible'];gd=p['global_discriminating'];gv=p['global_veto']
                for arm,refs,v,w in [('drop_increment_no_ref',[],visual,weak),('drop_increment_full',rr,visual,weak),('conservative_no_ref',[],min(delta)>=.5,min(delta)>=0),('conservative_full',rr,min(delta)>=.5,min(delta)>=0)]:flags[arm],strength[arm]=decision(v,w,gd,gv,refs)
                diag.append({'pair_alias':pair['pair_alias'],'qB_PRIVATE':pair['qB_PRIVATE'],'flags':flags,'relation_counts':strength,'mean_full_delta':p['mean_full_delta'],'relations':rr})
            for arm in ARMS[3:]:
                okay=[p for p in diag if p['flags'][arm]]
                if okay:sel[arm]=max(okay,key=lambda p:(p['relation_counts'][arm],p['mean_full_delta'],-p['qB_PRIVATE']))['qB_PRIVATE']
        ix=np.array(list(sel.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(sel)),query_indices=ix,arm_boxes_raw6=z['boxes'][ix]);dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'alias':r['alias'],'pairs':diag})
    seal(out,'R22_increment_veto_removed_cache_controls_before_new_scores',n=208,arms=ARMS,optimizer_steps=0);print('CACHE22_SEALED')
if __name__=='__main__':main()
