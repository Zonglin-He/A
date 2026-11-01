"""Factor controls: appearance gate, composed references, disjoint roles, frame scope.
No task annotation imports; output sealed before CPU task scoring.
"""
import numpy as np
from common21 import *
from observations import load
from assemble_v2 import relation_state
from detect import rect_iou
ARMS=['source','r20_fullG','gate_repair_no_reference','direct_full_relation_G','direct_no_role_exclusion','direct_raw_reference_proposals','direct_strict_frame_only','direct_relation_only','direct_drop_increment_only','direct_crop_repair_keep_increment']
def relation_evidence(g,scene,d,reference,pair,exclude=True,raw=False,strict=False):
    W,H=scene['image_size_wh'];aa,bb=pair['mask_indices_PRIVATE'];A,B=scene['objects'][aa],scene['objects'][bb];rr=[]
    for rel in g['reference_relations']:
        eid=rel['subject'];refs=reference['references'].get(eid,[])
        if raw:
            det=d['objects'][eid];refs=[{'box_PRIVATE':det['original']['boxes_PRIVATE'][j],'semantic_accepted':True} for j in det['stable_indices'][:3]]
        both=[]
        for obj in (A,B):
            boxes=[];excluded=0
            for obs in refs:
                if not obs['semantic_accepted']:continue
                # Enforce role inequality under EACH target hypothesis, never assume q0 is true.
                if exclude and rect_iou(obs['box_PRIVATE'],obj['detector_box_xyxy'])>=.3:excluded+=1;continue
                boxes.append(np.asarray(obs['box_PRIVATE'])/np.array([W,H,W,H]))
            rel2=dict(rel)
            if strict and rel['frame_provenance']!='explicit_clause':rel2['frame']='UNKNOWN'
            state=relation_state(rel2,obj,boxes);state['same_role_boxes_excluded']=excluded;both.append(state)
        rr.append({'entity':eid,'predicate':rel['predicate'],'frame_provenance':rel['frame_provenance'],'A':both[0],'B':both[1],'edge_supports_B_over_A':both[0]['status']=='CONTRADICT' and both[1]['status']=='SUPPORT'})
    return rr
def decision(visual,weak,globaldisc,globalveto,rr,relation_only=False):
    veto=any(x['B']['status']=='CONTRADICT' for x in rr);discrim=any(x['edge_supports_B_over_A'] for x in rr)
    accepted=(False if relation_only else visual) or (discrim or (globaldisc and not relation_only)) and weak
    return bool(accepted and not globalveto and not veto),sum(x['edge_supports_B_over_A'] for x in rr)
def main():
    lock('ASSEMBLE_v3_LOCK.json');verify(BASE/'reference208_attempt003');out=BASE/'direct208_attempt003';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();(out/'predictions_PRIVATE').mkdir()
    for r in read(ROWS):
        z=feature(r);q0=int(z['scores'].argmax());selected={a:q0 for a in ARMS};oldcase=read(ROOT/'crossplatform/round20/direct208_attempt002/cases_PRIVATE'/(r['id']+'.json'))
        with np.load(ROOT/'crossplatform/round20/direct208_attempt002/predictions_PRIVATE'/(r['id']+'.npz')) as x:selected['r20_fullG']=int(x['query_indices'][x['arm_names'].tolist().index('direct_fullG')])
        g=read(BASE/'text208_attempt003/cases_PRIVATE'/(r['id']+'.json'));d=read(BASE/'dino208_attempt003/cases_PRIVATE'/(r['id']+'.json'));ref=read(BASE/'reference208_attempt003/cases_PRIVATE'/(r['id']+'.json'));prep=old(r,'pairs');diag=[]
        if prep['pairs']:
            frame,scene,binding=load(r)
            for pair,prev in zip(prep['pairs'],oldcase['pair_diagnostics']):
                assert pair['pair_alias']==prev['pair_alias'];delta=prev['full_delta'];visual=min(delta)>=-.5 and float(np.mean(delta))>=.5;weak=min(delta)>=-.5;globalrr=[x for x in prev['relations'] if not x['is_reference_relation']];globaldisc=any(x['A']['status']=='CONTRADICT' and x['B']['status']=='SUPPORT' for x in globalrr);globalveto=any(x['B']['status']=='CONTRADICT' for x in globalrr);flags={};counts={};edges={}
                for arm,kw in [('gate_repair_no_reference',None),('direct_full_relation_G',{}),('direct_no_role_exclusion',{'exclude':False}),('direct_raw_reference_proposals',{'raw':True}),('direct_strict_frame_only',{'strict':True}),('direct_relation_only',{})]:
                    rr=relation_evidence(g,scene,d,ref,pair,**kw) if kw is not None else [];okay,n=decision(visual,weak,globaldisc,globalveto,rr,relation_only=arm=='direct_relation_only');flags[arm]=okay;counts[arm]=n;edges[arm]=rr
                flags['direct_drop_increment_only']=bool((min(delta)>=.5 or globaldisc and min(delta)>=0) and not globalveto)
                flags['direct_crop_repair_keep_increment']=bool((visual and min(prev['attribute_delta'])>=0 or globaldisc and weak) and not globalveto)
                for arm in ['direct_drop_increment_only','direct_crop_repair_keep_increment']:counts[arm]=0;edges[arm]=[]
                diag.append({'pair_alias':pair['pair_alias'],'qB_PRIVATE':pair['qB_PRIVATE'],'full_delta':delta,'old_attribute_delta':prev['attribute_delta'],'visual_gate_repaired':visual,'weak_visual_compatible':weak,'global_discriminating':globaldisc,'global_veto':globalveto,'flags':flags,'discriminating_counts':counts,'edges':edges,'mean_full_delta':float(np.mean(delta))})
            for arm in ARMS[2:]:
                accepted=[p for p in diag if p['flags'][arm]]
                if accepted:selected[arm]=max(accepted,key=lambda p:(p['discriminating_counts'][arm],p['mean_full_delta'],-p['qB_PRIVATE']))['qB_PRIVATE']
        indices=np.array(list(selected.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(selected)),query_indices=indices,arm_boxes_raw6=z['boxes'][indices]);dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'alias':r['alias'],'pair_diagnostics':diag,'reference_count':len(g['references']),'parameter_updates':0})
    seal(out,'R21_all208_factorial_direct_predictions_sealed_before_current_GT',n=208,arms=ARMS,optimizer_steps=0);print('DIRECT21_SEALED',208,len(ARMS))
if __name__=='__main__':main()
