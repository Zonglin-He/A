"""Matched no-reference and full-graph direct use with explicit pair uncertainty."""
import numpy as np
from common22 import *
from cache22 import refs_without_increment
from observations import load
from assemble21_v3 import relation_evidence
from depth22 import context as depth_context,evidence as depth_evidence,combine
ARMS=['source','r21_no_ref','r21_full','drop_increment_no_ref','drop_increment_full','conservative_no_ref','conservative_full','direct_pair_no_ref','direct_pair_full','direct_pair_no_role_conflict','direct_pair_relation_tiebreak','direct_pair_depth_full','direct_pair_depth_tiebreak']
def preference(result,margin=.5):
    values={'A':[],'B':[]};unclear=[]
    for view in result['views']:
        blank=view['blank']['logits']
        for order in ['normal','reversed']:
            v=view[order]['logits'];mapping={'A':'A','B':'B'} if order=='normal' else {'A':'B','B':'A'}
            for candidate in ['A','B']:
                own=mapping[candidate];other=mapping['B' if candidate=='A' else 'A'];raw=v[own]-v[other];blankmargin=blank[own]-blank[other];values[candidate].append({'raw_pair_margin':raw,'vs_uncertain_margin':v[own]-v['U'],'image_minus_blank_margin':raw-blankmargin})
    supports={c:all(min(v.values())>=margin for v in xx) for c,xx in values.items()};label='A' if supports['A'] else 'B' if supports['B'] else 'U'
    return {'label':label,'A_margins':values['A'],'B_margins':values['B'],'strength_B':min(min(x.values()) for x in values['B'])}
def main():
    lock('ASSEMBLE_LOCK.json');verify(BASE/'pair208_attempt001');verify(BASE/'cache208_attempt001');out=BASE/'direct208_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();(out/'predictions_PRIVATE').mkdir()
    for row_index,r in enumerate(read(ROWS)):
        z=feature(r);sel={a:int(z['scores'].argmax()) for a in ARMS}
        with np.load(BASE/'cache208_attempt001/predictions_PRIVATE'/(r['id']+'.npz')) as cached:
            for name,q in zip(cached['arm_names'].tolist(),cached['query_indices']):sel[name]=int(q)
        pairs=old(r,'pairs')['pairs'];vision=read(BASE/'pair208_attempt001/cases_PRIVATE'/(r['id']+'.json'));oldcase=prior(r,'direct');g=prior(r,'text');d=prior(r,'dino');ref=refs_without_increment(r);diag=[]
        if pairs:
            frame,scene,binding=load(r);depthctx=depth_context(r,scene,z);assert len(pairs)==len(vision['pairs'])==len(oldcase['pair_diagnostics'])
            for pair,pv,pc in zip(pairs,vision['pairs'],oldcase['pair_diagnostics']):
                assert pair['pair_alias']==pv['pair_alias']==pc['pair_alias'];subject=preference(pv['subject']);roles={eid:preference(v) for eid,v in pv['reference_roles'].items()};rr=relation_evidence(g,scene,d,ref,pair);disc=any(x['edge_supports_B_over_A'] for x in rr);veto=any(x['B']['status']=='CONTRADICT' for x in rr);roleconflict=subject['label']=='B' and any(x['label']=='B' for x in roles.values());weak=pc['weak_visual_compatible'];gd=pc['global_discriminating'];gv=pc['global_veto'];no_ref=weak and not gv and (subject['label']=='B' or gd and subject['label']!='A');full=weak and not gv and not veto and (subject['label']=='B' or (gd or disc) and subject['label']!='A');flags={'direct_pair_no_ref':bool(no_ref),'direct_pair_full':bool(full and not roleconflict),'direct_pair_no_role_conflict':bool(full)}
                dr=depth_evidence(r,g,scene,ref,pair,depthctx);combined=combine(rr,dr);ddisc=any(x['edge_supports_B_over_A'] for x in combined);dveto=any(x['B']['status']=='CONTRADICT' for x in combined);flags['direct_pair_depth_full']=bool(weak and not gv and not dveto and not roleconflict and (subject['label']=='B' or (gd or ddisc) and subject['label']!='A'))
                flags['direct_pair_relation_tiebreak']=bool(no_ref or weak and not gv and subject['label']!='A' and disc and not veto)
                flags['direct_pair_depth_tiebreak']=bool(no_ref or weak and not gv and subject['label']!='A' and ddisc and not dveto)
                diag.append({'pair_alias':pair['pair_alias'],'qB_PRIVATE':pair['qB_PRIVATE'],'subject_preference':subject,'reference_role_preferences':roles,'role_conflict':bool(roleconflict),'reference_discriminating':bool(disc),'reference_veto':bool(veto),'depth_discriminating':bool(ddisc),'depth_veto':bool(dveto),'global_discriminating':gd,'global_veto':gv,'weak_crop_compatible':weak,'relations':rr,'depth_edges':dr,'combined_edges':combined,'flags':flags,'relation_support_count':sum(x['edge_supports_B_over_A'] for x in rr),'depth_support_count':sum(x['edge_supports_B_over_A'] for x in combined)})
            for arm in ARMS[7:]:
                okay=[p for p in diag if p['flags'][arm]]
                if okay:sel[arm]=max(okay,key=lambda p:((p['depth_support_count'] if 'depth' in arm else p['relation_support_count'] if arm!='direct_pair_no_ref' else 0),p['subject_preference']['strength_B'],-p['qB_PRIVATE']))['qB_PRIVATE']
            # Tie-breaking cannot overwrite an existing no-reference decision with an
            # unverified reference. A clear role conflict means abstention, not a new ID.
            for arm in ['direct_pair_relation_tiebreak','direct_pair_depth_tiebreak']:
                if sel['direct_pair_no_ref']!=int(z['scores'].argmax()):
                    chosen=next(p for p in diag if p['qB_PRIVATE']==sel['direct_pair_no_ref']);sel[arm]=int(z['scores'].argmax()) if chosen['role_conflict'] else sel['direct_pair_no_ref']
                elif sel[arm]!=int(z['scores'].argmax()):
                    chosen=next(p for p in diag if p['qB_PRIVATE']==sel[arm])
                    if chosen['role_conflict']:sel[arm]=int(z['scores'].argmax())
        ix=np.array(list(sel.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(sel)),query_indices=ix,arm_boxes_raw6=z['boxes'][ix]);dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'alias':r['alias'],'pairs':diag,'actual_TTA_steps':0})
        if (row_index+1)%16==0:print('ASSEMBLE22',row_index+1,flush=True)
    seal(out,'R22_real_pair_no_increment_full_and_no_reference_sealed',n=208,arms=ARMS,optimizer_steps=0);print('DIRECT22_SEALED')
if __name__=='__main__':main()
