"""Independent pure CPU references for R49; no dataset or model access at import."""
import ast
import numpy as np

def extract_original_relation(source_rank23,source_detect20):
    """Execute only frozen pure function ASTs, never their I/O imports/main."""
    namespace={'np':np}
    for source,names in [(source_detect20,{'rect_iou'}),(source_rank23,{'sigmoid','relation_value'})]:
        tree=ast.parse(source);nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
        assert {n.name for n in nodes}==names
        for n in nodes:assert not any(isinstance(x,(ast.Import,ast.ImportFrom)) for x in ast.walk(n))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-original-pure-reference>','exec'),namespace)
    return namespace['relation_value']

def image_only_scene(instances):
    height,width=map(int,instances['original_hw']);masks=np.asarray(instances['masks_bool']);boxes=np.asarray(instances['boxes_xyxy'],dtype=np.float64)
    assert masks.dtype==np.bool_ and masks.shape==(len(boxes),height,width)
    items=[]
    for k in range(len(masks)):
        coordinates=np.argwhere(masks[k]);center=(coordinates[:,::-1].mean(0)+.5)/np.array([width,height]) if len(coordinates) else None
        edge=bool(len(coordinates) and (masks[k,0].any() or masks[k,-1].any() or masks[k,:,0].any() or masks[k,:,-1].any()))
        items.append({'mask_centroid_norm2':center,'touches_image_border':edge,'detector_box_xyxy':boxes[k]})
    return width,height,items

def operator(source,W,e):
    active=W.sum(axis=1)>0;cue=W@e;first=int(source.argmax());a=cue[first] if active[first] else float(np.median(e));raw=np.zeros(256);raw[active]=(cue[active]-a)/2;out=np.zeros(256);out[active]=np.clip(raw[active],-2,2)
    return active,cue,a,raw,out

def reference_stage(source,W,pool,evidence,text,instances,relation_function):
    s=np.asarray(source);assert s.shape==(256,) and W.shape==(256,len(pool));v=np.zeros(256);p=v.copy();r=v.copy()
    if not pool:return {'W':W,'mapped':np.zeros(256,bool),'image_e':np.zeros(0),'image_full':np.zeros(0),'image_gray':np.zeros(0),'vv':v.copy(),'rr':np.zeros((256,2)),'mask_relations':np.zeros((0,2)),'vision_unclipped':v.copy(),'position_unclipped':v.copy(),'relation_unclipped':v.copy(),'vision':v,'position':p,'relation':r,'anchor_v':None,'anchor_r':None,'F':s.astype(float)}
    e=np.array([np.mean([view['visual_evidence'] for view in evidence['targets_PRIVATE'][str(m)]]) for m in pool]);full=np.array([np.mean([view['full']['log_odds_yes_no'] for view in evidence['targets_PRIVATE'][str(m)]]) for m in pool]);gray=np.array([np.mean([view['gray']['log_odds_yes_no'] for view in evidence['targets_PRIVATE'][str(m)]]) for m in pool]);width,height,objects=image_only_scene(instances);per_mask=np.array([relation_function(text,objects[m],evidence,width,height) for m in pool]);active,cue,av,vu,v=operator(s,W,e);rr=W@per_mask;q0=int(s.argmax());ar=rr[q0] if active[q0] else np.median(per_mask,axis=0);pu=np.zeros(256);ru=np.zeros(256);pu[active]=rr[active,0]-ar[0];ru[active]=rr[active,1]-ar[1];p[active]=np.clip(pu[active],-2,2);r[active]=np.clip(ru[active],-2,2)
    return {'W':W,'mapped':active,'image_e':e,'image_full':full,'image_gray':gray,'vv':cue,'rr':rr,'mask_relations':per_mask,'vision_unclipped':vu,'position_unclipped':pu,'relation_unclipped':ru,'vision':v,'position':p,'relation':r,'anchor_v':av,'anchor_r':ar,'F':s.astype(float)+2*v+2*p+2*r}

ARMS=[f'legacy_V{v}P{p}R{r}' for v in (0,1) for p in (0,1) for r in (0,1)]+['legacy_rawvv_mapped_cue_only','legacy_postclipV_mapped_cue_only','legacy_fullG_mapped_only','legacy_fullG_V_unclipped','legacy_fullG_no_gray','joint_e_fullG_same_operator','joint_literal_source_plus_rawcue','joint_rawcue_mapped_only']
def reference_readouts(source,stage,joint_e):
    s=source.astype(float);v,p,r=[stage[k] for k in ['vision','position','relation']];W=stage['W'];active=stage['mapped']
    if len(joint_e):_,jcue,_,_,jv=operator(s,W,joint_e);_,_,_,_,fv=operator(s,W,stage['image_full'])
    else:jcue=np.zeros(256);jv=jcue.copy();fv=jcue.copy()
    scores=np.stack([s+2*a*v+2*b*p+2*c*r for a in (0,1) for b in (0,1) for c in (0,1)]+[stage['vv'],v,stage['F'],s+2*stage['vision_unclipped']+2*p+2*r,s+2*fv+2*p+2*r,s+2*jv+2*p+2*r,s+2*jcue,jcue]);domains=np.ones((16,256),bool);domains[[8,9,10,15]]=active;qs=[];ties=[];fallback=[]
    for i in range(16):
        eligible=np.flatnonzero(domains[i]);empty=len(eligible)==0;fallback.append(empty)
        if empty:qs.append(int(s.argmax()));ties.append([]);continue
        winners=eligible[scores[i,eligible]==scores[i,eligible].max()];qs.append(int(winners[s[winners].argmax()]) if i in [8,9,15] else int(winners[0]));ties.append(winners.tolist())
    return scores,domains,np.array(qs,np.int64),ties,fallback,jcue,jv,fv
