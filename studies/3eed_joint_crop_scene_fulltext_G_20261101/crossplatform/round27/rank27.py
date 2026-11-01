"""Bounded soft residual ranking. Unknown relations contribute zero, never veto."""
import numpy as np
from common27 import *
from observations27 import load
from assemble_v2 import relation_state
from detect import rect_iou
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def relation_value(g,obj,evidence,W,H):
    x=float(obj['mask_centroid_norm2'][0]);global_values=[];reference_values=[]
    if obj['touches_image_border']:return 0.,0.
    for rel in g['global_relations']:
        p=rel['predicate']
        if p in ['GLOBAL_LEFT','LEFT_OF_CENTER']:global_values.append(float(-np.tanh((x-.5)/.15)))
        elif p in ['GLOBAL_RIGHT','RIGHT_OF_CENTER']:global_values.append(float(np.tanh((x-.5)/.15)))
        elif p=='GLOBAL_CENTER':global_values.append(float(1-4*abs(x-.5)))
    for rel in g['reference_relations']:
        if rel['frame']!='camera_image':continue
        anchors=evidence['references'].get(rel['subject'],[]);weighted=0.;total=0.
        for a in anchors:
            if rect_iou(a['box_PRIVATE'],obj['detector_box_xyxy'])>=.3:continue
            box=np.asarray(a['box_PRIVATE'])/np.array([W,H,W,H]);center=(box[0]+box[2])/2;uncertainty=.1*(box[2]-box[0]);direction=1 if rel['predicate']=='RIGHT_OF' else -1;weight=float(sigmoid(a['full']['log_odds_yes_no'])*sigmoid(a['visual_evidence'])*a['detector_score']);score=float(np.tanh(direction*(center-x)/(.15+uncertainty)));weighted+=weight*score;total+=weight
        if total:reference_values.append(weighted/(1+total))
    return (float(np.mean(global_values)) if global_values else 0.,float(np.mean(reference_values)) if reference_values else 0.)
def arrays(r):
    with np.load(BASE/'preparedQ64_attempt001/arrays_PRIVATE'/(r['id']+'.npz')) as z:return {k:z[k].copy() for k in z.files}
def evidence_scores(r):
    z=feature(r);p=read(BASE/'preparedQ64_attempt001/cases_PRIVATE'/(r['id']+'.json'));g=read(BASE/'textQ64_attempt001/cases_PRIVATE'/(r['id']+'.json'));e=read(BASE/'evidenceQ64_attempt001/cases_PRIVATE'/(r['id']+'.json'));a=arrays(r);pool=p['mask_pool_PRIVATE'];q0=int(z['scores'].argmax());vs=np.zeros(256);rs=np.zeros(256);ps=np.zeros(256);baseline=q0;legacy=p['legacy_preparation_PRIVATE'];diag=[]
    if pool:
        frame,scene,binding=load(r);W,H=scene['image_size_wh'];ev=np.array([np.mean([x['visual_evidence'] for x in e['targets_PRIVATE'][str(m)]]) for m in pool]);rv=np.array([relation_value(g,scene['objects'][m],e,W,H) for m in pool]);weights=a['weights'];mapped=weights.sum(1)>0;vv=weights@ev;rr=weights@rv;anchor_v=vv[q0] if mapped[q0] else float(np.median(ev));anchor_r=rr[q0] if mapped[q0] else np.median(rv,axis=0);vs[mapped]=np.clip((vv[mapped]-anchor_v)/2,-2,2);ps[mapped]=np.clip(rr[mapped,0]-anchor_r[0],-2,2);rs[mapped]=np.clip(rr[mapped,1]-anchor_r[1],-2,2)
        # Exact R21 no-reference readout on the original <=2 candidate pairs.
        for pair in legacy['pairs']:
            aa,bb=pair['mask_indices_PRIVATE'];A,B=scene['objects'][aa],scene['objects'][bb];delta=[y['visual_evidence']-x['visual_evidence'] for x,y in zip(e['targets_PRIVATE'][str(aa)],e['targets_PRIVATE'][str(bb)])];states=[(relation_state(rel,A,[]),relation_state(rel,B,[])) for rel in g['global_relations']];disc=any(x['status']=='CONTRADICT' and y['status']=='SUPPORT' for x,y in states);veto=any(y['status']=='CONTRADICT' for x,y in states);okay=((min(delta)>=-.5 and np.mean(delta)>=.5) or disc and min(delta)>=-.5) and not veto;diag.append({'qB_PRIVATE':pair['qB_PRIVATE'],'mean_delta':float(np.mean(delta)),'accepted':bool(okay)})
        accepted=[x for x in diag if x['accepted']]
        if accepted:baseline=max(accepted,key=lambda x:(x['mean_delta'],-x['qB_PRIVATE']))['qB_PRIVATE']
    return {'source':q0,'R21_no_ref':int(baseline),'vision':vs,'position':ps,'relation':rs,'token':a['token_delta'],'component':a['component'],'legacy':legacy,'preparation':p}
def soft_score(z,e,cfg,vision=True,relation=True,token=True,restricted=False):
    v=e['vision'].copy();rel=e['relation'].copy();pos=e['position'].copy()
    if restricted:
        allowed={int(z['scores'].argmax())}|{p['qB_PRIVATE'] for p in e['legacy']['pairs']};mask=np.ones(256,bool);mask[list(allowed)]=False;v[mask]=0;rel[mask]=0;pos[mask]=0
    return np.log(np.maximum(z['scores'],1e-12))+cfg['gamma']*e['token']*token+cfg['vision']*v*vision+cfg['relation']*(pos+rel*relation)
def main():
    lock('RANK_LOCK.json');verify(BASE/'evidenceQ64_attempt001');out=BASE/'bankQ64_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir()
    for i,r in enumerate(read(EXP.parent/'input/Q64_PRIVATE.json')):
        z=feature(r);e=evidence_scores(r);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),vision=e['vision'],position=e['position'],relation=e['relation'],token=e['token'],component=e['component'],source=np.array(e['source']),R21_no_ref=np.array(e['R21_no_ref']));dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'alias':r['alias'],'legacy_PRIVATE':e['legacy'],'objects':e['preparation']['objects']})
        if (i+1)%16==0:print('BANK23',i+1,flush=True)
    seal(out,'R23_score_bank_all272_GT_free_no_hard_evidence_veto',n=64,optimizer_steps=0);print('BANK23_SEALED')
if __name__=='__main__':main()
