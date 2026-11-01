"""Conservative observer-depth relation intervals, never surface-as-box fitting."""
import sys,re,numpy as np
from common22 import *
sys.path.insert(0,str(ROOT/'crossplatform/round17'))
from pair_geometry import prepare
from camera_projection import model_to_camera_matrix
from detect import rect_iou
def context(r,scene,z):
    raw=prepare(scene,z['boxes'],r['id']);support=np.asarray(raw['point_support_count']);coverage=np.asarray(raw['point_support_fraction']);pi=np.asarray(raw['projection_iou']);row=np.abs(model_to_camera_matrix(scene['camera_spec'])[2,:3]);intervals=[]
    for m,obj in enumerate(scene['objects']):
        depth=np.asarray(scene['depth_camera'],float)[np.asarray(obj['exclusive_visible_indices'],int)];qs=np.flatnonzero((support[:,m]>=3)&(coverage[:,m]>=.15)&(pi[:,m]>=.1)&(z['boxes'][:,3:]>0).all(1))
        if len(depth)<3 or not len(qs) or obj['n_exclusive']<.8*max(obj['n_visible'],1):intervals.append(None);continue
        q=max(qs,key=lambda k:(float(z['scores'][k]),-int(k)));extent=float(row@z['boxes'][q,3:]);lo,hi=np.quantile(depth,[.1,.9]);intervals.append({'lo':float(lo-extent),'hi':float(hi+extent),'visible_depth_quantiles':[float(lo),float(hi)],'complete_extent_prior_padding_m':extent,'point_count':len(depth),'not_full_object_boundary':True})
    return {'intervals':intervals}
def edges(r,g):
    output=[];caption=r['caption']
    for i,ref in enumerate(g['references']):
        end=g['references'][i+1]['span'][0] if i+1<len(g['references']) else len(caption);tail=caption[ref['span'][1]:end];tail=re.split(r'[.;]',tail)[0];front=bool(re.search(r'\bin\s+front\b',tail,re.I));back=bool(re.search(r'\bbehind\b',tail,re.I))
        if front==back:continue
        body=bool(re.search(r'\b(?:its|their|his|her)\s+(?:own\s+)?(?:front|back|left|right)\b',tail,re.I));camera=bool(re.search(r'\b(view|image|picture|frame|observer)\b',caption,re.I));output.append({'entity':ref['entity'],'predicate':'REFERENCE_BEHIND_TARGET' if back else 'REFERENCE_IN_FRONT_OF_TARGET','frame':'observer_camera_context' if camera and not body else 'UNKNOWN','frame_is_context_assumption':True})
    return output
def evidence(r,g,scene,reference,pair,depthctx):
    result=[];A,B=pair['mask_indices_PRIVATE']
    for edge in edges(r,g):
        statuses=[]
        for target in [A,B]:
            ti=depthctx['intervals'][target];states=[];reasons=[]
            for obs in reference['references'].get(edge['entity'],[]):
                if not obs['semantic_accepted']:continue
                if rect_iou(obs['box_PRIVATE'],scene['objects'][target]['detector_box_xyxy'])>=.3:continue
                matches=sorted([(rect_iou(obs['box_PRIVATE'],o['detector_box_xyxy']),i) for i,o in enumerate(scene['objects'])],reverse=True)
                if not matches or matches[0][0]<.5 or len(matches)>1 and matches[0][0]-matches[1][0]<.1:states.append('UNKNOWN');reasons.append('reference_mask_mapping_not_unique');continue
                m=matches[0][1];ri=depthctx['intervals'][m]
                if m==target or ti is None or ri is None or edge['frame']=='UNKNOWN':states.append('UNKNOWN');reasons.append('depth_or_coordinate_frame_unobservable');continue
                behind=ri['lo']>ti['hi']+.5;front=ri['hi']<ti['lo']-.5
                s='SUPPORT' if (behind if edge['predicate']=='REFERENCE_BEHIND_TARGET' else front) else 'CONTRADICT' if (front if edge['predicate']=='REFERENCE_BEHIND_TARGET' else behind) else 'UNKNOWN';states.append(s);reasons.append('visible_depth_intervals_with_complete_extent_prior')
            status=states[0] if states and len(set(states))==1 else 'UNKNOWN';statuses.append({'status':status,'anchor_states':states,'reasons':reasons})
        result.append(dict(edge,A=statuses[0],B=statuses[1],edge_supports_B_over_A=statuses[0]['status']=='CONTRADICT' and statuses[1]['status']=='SUPPORT'))
    return result
def combine(image_edges,depth_edges):
    output=[];used=set()
    for edge in image_edges:
        d=next((d for d in depth_edges if d['entity']==edge['entity']),None)
        if d is None:output.append(edge);continue
        used.add(edge['entity']);c=dict(edge,depth=d,conjunctive_image_and_depth=True)
        for target in ['A','B']:
            states=[edge[target]['status'],d[target]['status']];c[target]=dict(edge[target],status='CONTRADICT' if 'CONTRADICT' in states else 'SUPPORT' if all(s=='SUPPORT' for s in states) else 'UNKNOWN')
        c['edge_supports_B_over_A']=c['A']['status']=='CONTRADICT' and c['B']['status']=='SUPPORT';output.append(c)
    for d in depth_edges:
        if d['entity'] not in used:output.append(d)
    return output
