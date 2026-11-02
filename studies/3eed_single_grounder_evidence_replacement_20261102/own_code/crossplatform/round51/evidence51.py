"""Pure R51 evidence/readout functions; no files, models, GT or external cache.

Surface support is used only for association and visible image position, never
as complete object extent. Unknown attribute/relationship evidence is zero.
"""
import numpy as np


def attr_evidence(attributes, entity='T'):
    requests={r['entity']:r for r in attributes['entity_attribute_requests']}
    words=requests.get(entity,{}).get('recognized_color_words',[])
    values=[];known=[]
    for item in attributes['instances']:
        observed=[item['color_logodds_relative_scene_visible'].get(c) for c in words]
        observed=[v for v in observed if v is not None and np.isfinite(v)]
        known.append(bool(observed))
        values.append(float(np.mean(observed))*item['color_evidence_reliability'] if observed else 0.)
    return np.asarray(values,float),np.asarray(known,bool)


def object_geometry(a, image_size_wh):
    W,H=image_size_wh;out=[]
    for j,rect in enumerate(a['instance_rect_xyxy']):
        lo,hi=a['instance_crop_mask_pixel_offsets'][j:j+2];xy=a['instance_crop_mask_pixel_xy'][lo:hi]
        defined=len(xy)>0
        out.append({'visible':defined,'domain':'raw3d','instance_index':j,'xy':(xy.mean(0)+.5)/np.array([W,H]) if defined else None,
                    'rect':np.asarray(rect,float),'border':bool(defined and ((xy[:,0]==0)|(xy[:,1]==0)|(xy[:,0]==W-1)|(xy[:,1]==H-1)).any())})
    return out


def raw_relations(graph, objects, references, image_size_wh):
    """Same image-left/right direction as original typed-reference operator.

references[entity] has {xy_norm2, width_norm, probability} hypotheses. Their
probability mass is not renormalized after missing/truncated observations.
"""
    pr=np.zeros((len(objects),2),float);known=np.zeros((len(objects),2),bool)
    for j,obj in enumerate(objects):
        if not obj['visible'] or obj['border']:continue
        x=float(obj['xy'][0]);glob=[];rel=[]
        for g in graph.get('global_relations',[]):
            if g.get('frame')!='camera_image':continue
            p=g['predicate']
            if p in ('GLOBAL_LEFT','LEFT_OF_CENTER'):glob.append(float(-np.tanh((x-.5)/.15)))
            elif p in ('GLOBAL_RIGHT','RIGHT_OF_CENTER'):glob.append(float(np.tanh((x-.5)/.15)))
            elif p=='GLOBAL_CENTER':glob.append(1-4*abs(x-.5))
        if glob:pr[j,0]=np.mean(glob);known[j,0]=True
        for g in graph.get('reference_relations',[]):
            if g.get('frame')!='camera_image' or g.get('predicate') not in ('LEFT_OF','RIGHT_OF'):continue
            hs=references.get(g['subject'],[]);total=0.;score=0.
            for h in hs:
                if h.get('xy_norm2') is None:continue
                if h.get('instance_domain')==obj.get('domain') and h.get('instance_index')==obj.get('instance_index'):continue
                # Same target/reference overlap exclusion as original R23.
                if h.get('rect') is not None:
                    aa=np.asarray(h['rect'],float);bb=np.asarray(obj['rect'],float)
                    inter=np.maximum(np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2]),0).prod()
                    union=np.maximum(aa[2:]-aa[:2],0).prod()+np.maximum(bb[2:]-bb[:2],0).prod()-inter
                    if union>0 and inter/union>=.3:continue
                weight=float(h['probability'])
                if weight<=0:continue
                direction=1 if g['predicate']=='RIGHT_OF' else -1
                score+=weight*float(np.tanh(direction*(h['xy_norm2'][0]-x)/(.15+.1*h.get('width_norm',0.))))
                total+=weight
            if total>0:
                # Preserve missing-reference mass rather than converting a
                # tiny supported hypothesis into fully certain supervision.
                explicit=any(h.get('uncertainty_mass_explicit',False) for h in hs)
                rel.append(score/(max(1.,total) if explicit else total));known[j,1]=True
        if rel:pr[j,1]=np.mean(rel)
    return pr,known


def propagate(source,W,e,pr):
    """Original native-score + 2V+2P+2R anchoring, clip and addition order."""
    s=np.asarray(source);W=np.asarray(W,float);e=np.asarray(e,float);pr=np.asarray(pr,float)
    assert s.shape==(256,) and W.shape==(256,len(e)) and pr.shape==(len(e),2)
    assert np.isfinite(e).all() and np.isfinite(pr).all()
    V=np.zeros(256);P=np.zeros(256);R=np.zeros(256);mapped=W.sum(1)>0
    if len(e):
        v=W@e;r=W@pr;q=int(s.argmax())
        av=v[q] if mapped[q] else float(np.median(e));ar=r[q] if mapped[q] else np.median(pr,axis=0)
        V[mapped]=np.clip((v[mapped]-av)/2,-2,2)
        P[mapped]=np.clip(r[mapped,0]-ar[0],-2,2);R[mapped]=np.clip(r[mapped,1]-ar[1],-2,2)
    return s.astype(float)+2*V+2*P+2*R,np.stack([V,P,R])


def propagate_measured(source,W,e,pr,e_known,pr_known):
    """Same scales, with explicit per-cue UNKNOWN neutrality after anchoring.

    Unknown instance values are zero contributions, never reassigned to known
    foreground. A query with NO support for a cue remains exactly zero for
    that cue after anchoring. A partially supported query keeps W@known-value
    without renormalizing its known mass. Legacy propagate above is unchanged.
    """
    s=np.asarray(source);W=np.asarray(W,float);e=np.asarray(e,float);pr=np.asarray(pr,float)
    ek=np.asarray(e_known,bool);pk=np.asarray(pr_known,bool)
    assert s.shape==(256,) and W.shape==(256,len(e)) and pr.shape==(len(e),2)
    assert ek.shape==(len(e),) and pk.shape==(len(e),2)
    assert np.isfinite(e).all() and np.isfinite(pr).all() and np.isfinite(W).all()
    values=np.c_[np.where(ek,e,0.),np.where(pk,pr,0.)]
    known=np.c_[ek,pk]
    q=int(s.argmax());cues=np.zeros((3,256));anchors=np.zeros(3)
    query_known=(W@known.astype(float))>0
    projected=W@values
    for c in range(3):
        if not known[:,c].any():continue
        anchor=projected[q,c] if query_known[q,c] else float(np.median(values[known[:,c],c]))
        anchors[c]=anchor
        valid=query_known[:,c]
        scale=2. if c==0 else 1.
        cues[c,valid]=np.clip((projected[valid,c]-anchor)/scale,-2,2)
    F=s.astype(float)+2*cues[0]+2*cues[1]+2*cues[2]
    return F,cues,{'query_known':query_known.T,'anchors':anchors,
                   'query_known_weight_mass':(W@known.astype(float)).T}


def reference_hypotheses(raw_xyz, instance_arrays, ref_boxes, ref_logodds, platform, attributes, entity, appearance_factor=1.):
    """Bind independent reference proposals to point-supported hypotheses.

Different phrase query indices are NEVER reused as target indices. Deduplicate
by geometric point-instance ID, max SOURCE score per ID; source-only top3 are
fixed before color. All-hypothesis+UNKNOWN0 source mass is preserved. Color may
reweight those same three hypotheses but cannot select another hypothesis or
renormalize away the excluded/unknown mass. No top1 identity is asserted.
"""
    a=instance_arrays;M=len(a['instance_rect_xyxy']);z=np.full(M,-np.inf)
    ref_boxes=np.asarray(ref_boxes);ref_logodds=np.asarray(ref_logodds)
    assert ref_boxes.shape==(256,6) and ref_logodds.shape==(256,)
    assert np.isfinite(ref_boxes).all() and np.isfinite(ref_logodds).all()
    xyz=np.asarray(raw_xyz,np.float32).copy()
    if platform=='drone':xyz[:,2]+=np.float32(1.8)
    for m in range(M):
        lo,hi=a['instance_point_offsets'][m:m+2];ids=a['instance_point_indices'][lo:hi]
        if len(ids)<6:continue
        points=xyz[ids];inside=(np.abs(points[None]-ref_boxes[:,None,:3])<=ref_boxes[:,None,3:]/2).all(-1)
        count=inside.sum(1);ok=(count>=3)&(count/max(len(ids),1)>=.15)&(ref_boxes[:,3:]>0).all(1)
        if ok.any():z[m]=np.max(ref_logodds[ok])
    finite=np.isfinite(z)
    if not finite.any():return [],{'unknown_probability':1.,'source_unknown_probability':1.,
        'source_truncated_probability':0.,'hypotheses':0,'retained_probability':0.,
        'source_top3_instance_ids':[],'source_hypothesis_logits':[None]*M,
        'source_probabilities':[0.]*M,'source_only_top3_fixed_before_color':True}
    mx=max(0.,float(z[finite].max()));mass=np.zeros(M);mass[finite]=np.exp(np.clip(z[finite]-mx,-700,0));unknown=np.exp(-mx);den=unknown+mass.sum();mass/=den
    keep=sorted(np.flatnonzero(finite),key=lambda j:(-float(mass[j]),int(j)))[:3]
    source_unknown=float(unknown/den)
    source_other=float(max(0.,1-source_unknown-sum(mass[j] for j in keep)))
    attr,known=attr_evidence(attributes,entity)
    adjusted=np.array([mass[j]*np.exp(np.clip(appearance_factor*attr[j],-8,8)) if known[j] else mass[j] for j in keep])
    newden=source_unknown+source_other+float(adjusted.sum())
    adjusted/=newden
    result=[{'instance_index':int(j),'probability':float(p),
             'source_probability':float(mass[j]),
             'attribute_known':bool(known[j]),
             'xy_pixel':a['instance_visible_centroid_uv'][j].tolist() if np.isfinite(a['instance_visible_centroid_uv'][j]).all() else None,
             'rect':a['instance_rect_xyxy'][j].tolist()} for j,p in zip(keep,adjusted)]
    return result,{'unknown_probability':float((source_unknown+source_other)/newden),
        'source_unknown_probability':source_unknown,'source_truncated_probability':source_other,
        'hypotheses':int(finite.sum()),'retained_probability':float(adjusted.sum()),
        'source_top3_instance_ids':[int(j) for j in keep],
        'source_hypothesis_logits':[float(x) if np.isfinite(x) else None for x in z],
        'source_probabilities':mass.tolist(),'source_only_top3_fixed_before_color':True,
        'unobservable_hypotheses_keep_mass_but_no_spatial_evidence':True}
