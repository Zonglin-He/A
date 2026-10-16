"""Independent saved-selection replay, then postseal geometry diagnostics.

Token geometry is a displayed-area diagnostic, not semantic ground truth.
Only the Source-coordinate control performs a new Student update in phase one.
"""
from pathlib import Path
import collections,gc,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,CONTROL,PRIOR,read,put,sha,verify
def control_score():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    verify('SOURCE');b=CONTROL/'trials/SOURCE';root=read(b/'ROOT_READBACK.json');bar=read(b/'GLOBAL_PREDICTION_BARRIER.json')
    assert root['status']=='pass'and root['count']==32 and bar['status']=='complete'and not bar['GT_read']
    old=read(PRIOR/'EXPERIMENT_POSTSEAL_RESULTS.json');rr=read(CONTROL/'ROSTER_LOCK.json')['records'];refs=read(b/'REFERENCE_LOCK.json')['records'];original=read(PRIOR/'trials/E1/REFERENCE_LOCK.json')['records']
    cases={r['probe_alias']:load_case(r)[0]for r in rr};changed=0
    for ref,orig in zip(refs,original):
        assert ref['case_id']==orig['case_id'];assert len(ref['anchors'])==len(orig['anchors'])
        for a,o in zip(ref['anchors'],orig['anchors']):
            assert {k:v for k,v in a.items()if k!='box_cxcywh'}=={k:v for k,v in o.items()if k!='box_cxcywh'}
            box=np.asarray(a['box_cxcywh']);xy=np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]
            assert np.allclose(xy,cases[ref['case_id']]['native_boxes_xyxy_normalized'][a['position']],rtol=0,atol=2e-16);changed+=1
    put(CONTROL/'ROOT_MATCHED_SUPPORT.json',dict(status='pass',count=32,reference_count=changed,coordinates_only_changed=True,GT_read=False,root_sha256=sha(b/'ROOT_READBACK.json')))
    targets,gt=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];cost=[]
    oldidx={(r['case_id'],r['arm']):r for r in old['rows']}
    for item in bar['files']:
        cid=item['case_id'];assert sha(ROOT/item['path'])==item['sha256'];d=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);c=cases[cid]
        support=[w['source_grid_position']for w in read(PRIOR/'spatial/E1/proposals'/f'{cid}.json')['windows']]
        for name,key in [('Frozen','F'),('Temporal','T'),('Source_control','TS_M2')]:
            score,_=metric._score_student_arm(d['arms'][key],c,targets[cid],support,official,independent)
            if name!='Source_control':assert score==oldidx[cid,name]['metrics'],(cid,name)
            rows.append(dict(case_id=cid,dataset=d['dataset'],arm=name,metrics=score,GT_read=True))
        for a in ('E0','E1','E2'):rows.append(oldidx[cid,a])
        f=d['fits']['M2'];cost.append(dict(case_id=cid,dataset=d['dataset'],actual_fit=f is not None,accepted_steps=f.get('optimizer_steps',0)if f else 0,nonzero_parameter_update=any(not torch.equal(v.cpu(),f['final_state'][k].cpu())for k,v in f['initial_state'].items())if f else False,GT_read=False))
        del d;gc.collect()
    summary,paired=summary_rows(rows)
    put(CONTROL/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,cost=cost,GT_bindings=gt,GT_read=True,scope='same E1 selected support/weights, only reference coordinates switched to Source; source_macro Query one per parent',prediction_barrier_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json')))
    put(CONTROL/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True,result_sha256=sha(CONTROL/'POSTSEAL_RESULTS.json')))
    print(json.dumps([r for r in paired if r['metric']=='m_vIoU']))
def summary_rows(rows):
    summary=[];paired=[]
    for ds in ('hc2','vidstg'):
        idx={(r['case_id'],r['arm']):r['metrics']for r in rows if r['dataset']==ds};ids=sorted({c for c,a in idx});arms=sorted({a for c,a in idx})
        for a in arms:
            for key in idx[ids[0],a]:
                vv=[idx[c,a][key]for c in ids if idx[c,a][key]is not None];summary.append(dict(dataset=ds,arm=a,metric=key,valid=len(vv),unknown=len(ids)-len(vv),mean=math.fsum(vv)/len(vv)if vv else None))
        for a,b in [('Source_control','Temporal'),('E1','Source_control'),('Source_control','E0'),('E1','Temporal')]:
            for key in idx[ids[0],a]:
                vv={c:100*(idx[c,a][key]-idx[c,b][key])for c in ids if idx[c,a][key]is not None and idx[c,b][key]is not None};v=list(vv.values())
                paired.append(dict(dataset=ds,comparison=a+'_minus_'+b,metric=key,valid=len(v),unknown=len(ids)-len(v),mean_delta_pp=math.fsum(v)/len(v)if v else None,positive=sum(x>1e-6 for x in v),negative=sum(x< -1e-6 for x in v),neutral=sum(abs(x)<=1e-6 for x in v),per_source_delta_pp=vv))
    return summary,paired
def response_root():
    import numpy as np
    from scripts.stvg_referent_response_v1 import analyze
    verify();bar=read(BASE/'RESPONSE_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==16 and not bar['GT_read'];n=0
    for r in bar['records']:
        assert sha(ROOT/r['path'])==r['sha256'];a=np.load(ROOT/r['path'],allow_pickle=False);T,H,W=r['merged_grid'];assert a['features'].shape[:2]==(T,H*W)
        b=analyze(a['features'],a['query_embedding'],a['caption_embedding'],a['swapped_embedding'],H,W)
        for k,v in b.items():
            if k=='selections':
                for s,ix in v.items():assert np.array_equal(ix,a['selection_'+s]),(r['case_id'],s)
            else:assert np.array_equal(v,a[k]),(r['case_id'],k)
        assert r['grid'][1:]==[32,32]and [T,H,W]==[4,16,16]
        n+=1;a.close()
    put(BASE/'ROOT_RESPONSE_READBACK.json',dict(status='pass',count=16,clip_count=n,selections_exact=True,all_scalar_proxy_scores_exact=True,GT_read=False,barrier_sha256=sha(BASE/'RESPONSE_GLOBAL_BARRIER.json'),scope='independent replay from saved native projected vectors/embeddings; no separate visual-encoder execution'))
    put(BASE/'GLOBAL_SELECTION_BARRIER.json',dict(status='complete',count=16,clip_count=n,GT_read=False,root_sha256=sha(BASE/'ROOT_RESPONSE_READBACK.json'),response_barrier_sha256=sha(BASE/'RESPONSE_GLOBAL_BARRIER.json')))
def geometry():
    import numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    assert read(BASE/'GLOBAL_SELECTION_BARRIER.json')['status']=='complete';assert read(BASE/'ROOT_RESPONSE_READBACK.json')['status']=='pass'
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,gt=truth._truths_after_seal(cases)
    tm={cid:metric._truth_map(targets[cid])for cid in cases};jobs=read(BASE/'INPUT_BARRIER.json')['jobs'];jj={(j['case_id'],read(ROOT/j['metadata_path'])['candidate']['canonical_id']):j for j in jobs};ds={r['probe_alias']:r['dataset']for r in rr}
    frame_rows=[];pair_rows=[];clips=[]
    for item in read(BASE/'RESPONSE_GLOBAL_BARRIER.json')['records']:
        cid=item['case_id'];key=item['candidate_id'];c=cases[cid];j=jj[cid,key];m=read(ROOT/j['metadata_path']);a=np.load(ROOT/item['path'],allow_pickle=False);fids=m['metadata']['sampled_physical_frame_ids'];assert len(fids)==8
        width,height=c['input']['width'],c['input']['height'];scale=min(448/width,448/height);rw,rh=round(width*scale),round(height*scale);ox,oy=(448-rw)//2,(448-rh)//2
        H,W=item['merged_grid'][1:];yy,xx=np.meshgrid(np.arange(H),np.arange(W),indexing='ij');cells=np.stack([xx*28,yy*28,(xx+1)*28,(yy+1)*28],-1).reshape(-1,4)
        names=[s.removeprefix('selection_')for s in a.files if s.startswith('selection_')];hits=[];known=[]
        def overlap(box):
            box=np.asarray(box);lo=np.maximum(cells[:,:2],box[:2]);hi=np.minimum(cells[:,2:],box[2:]);return np.maximum(hi-lo,0).prod(1)
        for t in range(4):
            hh=[]
            for fid in fids[2*t:2*t+2]:
                known_gt=int(fid)in tm[cid];box=np.asarray(tm[cid][int(fid)])/np.array([width,height,width,height])if known_gt else None
                target=np.array([ox+box[0]*rw,oy+box[1]*rh,ox+box[2]*rw,oy+box[3]*rh])if known_gt else None
                area=overlap(target)if target is not None else None
                if area is not None:hh.append(area)
                source=np.array([np.interp(fid,c['frame_ids'],c['native_boxes_xyxy_normalized'][:,i])for i in range(4)]);sb=np.array([ox+source[0]*rw,oy+source[1]*rh,ox+source[2]*rw,oy+source[3]*rh]);sa=overlap(sb)
                for name in names:
                    idx=a['selection_'+name][t];rec=dict(case_id=cid,dataset=ds[cid],candidate_id=key,physical_frame_id=int(fid),merged_cell=t,strategy=name,known_GT=known_gt,source_overlap_fraction=float(sa[idx].sum()/(len(idx)*28*28)),target_occupancy=float(area[idx].sum()/(len(idx)*28*28))if area is not None else None,target_box_coverage=float(area[idx].sum()/max(1,(target[2]-target[0])*(target[3]-target[1])))if area is not None else None,GT_read=True)
                    frame_rows.append(rec)
            hits.append(np.max(hh,axis=0)>0 if hh else None);known.append(bool(hh))
        for t in range(3):
            edges=[(int(i),int(k))for tt,i,k,v in a['edges']if int(tt)==t];valid=known[t]and known[t+1]
            for name in names:
                left=set(a['selection_'+name][t]);right=set(a['selection_'+name][t+1]);retained=[(i,k)for i,k in edges if i in left and k in right]
                support=sum(bool(hits[t][i]and hits[t+1][k])for i,k in retained)if valid else None
                pair_rows.append(dict(case_id=cid,dataset=ds[cid],candidate_id=key,transition=t,strategy=name,known_GT=valid,retained_proxy_edges=len(retained),supported_edges=support,adjacent_supported_pair_coverage=float(support>0)if valid else None,GT_read=True))
        clips.append(dict(case_id=cid,candidate_id=key,known_sampled_frames=sum(fid in tm[cid]for fid in fids),unknown_sampled_frames=sum(fid not in tm[cid]for fid in fids),empty_caption=item['caption_empty'],proxy_edges=len(a['edges']),cut_transitions=int(a['cuts'].sum())))
        a.close()
    source=[]
    for r in rr:
        cid=r['probe_alias']
        for s in names:
            ff=[x for x in frame_rows if x['case_id']==cid and x['strategy']==s];pp=[x for x in pair_rows if x['case_id']==cid and x['strategy']==s];record=dict(case_id=cid,dataset=r['dataset'],strategy=s)
            for k in ('target_occupancy','target_box_coverage','source_overlap_fraction'):
                # Average repeated candidate exposures within physical frame first.
                cells=collections.defaultdict(list)
                for x in ff:
                    if x[k]is not None:cells[x['physical_frame_id']].append(x[k])
                vals=[math.fsum(v)/len(v)for v in cells.values()];record[k]=math.fsum(vals)/len(vals)if vals else None;record[k+'_known_positions']=len(vals)
            vals=[x['adjacent_supported_pair_coverage']for x in pp if x['adjacent_supported_pair_coverage']is not None];record['adjacent_supported_pair_coverage']=math.fsum(vals)/len(vals)if vals else None;record['known_candidate_transitions']=len(vals);source.append(record)
    summary=[];gate=[]
    for domain in ('hc2','vidstg'):
        group=[x for x in source if x['dataset']==domain];means={}
        for s in names:
            for k in ('target_occupancy','target_box_coverage','source_overlap_fraction','adjacent_supported_pair_coverage'):
                vals=[x[k]for x in group if x['strategy']==s and x[k]is not None];v=math.fsum(vals)/len(vals)if vals else None;means[s,k]=v;summary.append(dict(dataset=domain,strategy=s,metric=k,valid=len(vals),unknown=8-len(vals),mean=v))
        occ=means['consensus','target_occupancy'];single=max(v for v in [means['query','target_occupancy'],means['caption','target_occupancy']]if v is not None);delta=occ-single if occ is not None else None
        connect=means['connection','adjacent_supported_pair_coverage'];query=means['query','adjacent_supported_pair_coverage'];gap=connect-query if connect is not None and query is not None else None
        n=sum(x['strategy']=='consensus'and x['target_occupancy']is not None for x in group);gate.append(dict(dataset=domain,known_sources=n,consensus_minus_best_single=delta,connection_minus_query=gap,pass_gate=n>=4 and delta is not None and delta>0 and gap is not None and gap>0))
    put(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json',dict(status='complete',count=16,source_rows=source,frame_rows=frame_rows,pair_rows=pair_rows,clips=clips,summary=summary,GT_bindings=gt,GT_read=True,scope='native projected feature proxy; target rectangle occupancy is NOT semantic identity/attention ER or CS/causal correction',barrier_sha256=sha(BASE/'GLOBAL_SELECTION_BARRIER.json')))
    put(BASE/'PHASE2_DECISION.json',dict(status='eligible_pending_native_pruning_qualification'if all(g['pass_gate']for g in gate)else'not_supported_stop_this_expansion',count=16,gate=gate,GT_read=True,rule_sha256=sha(BASE/'DESIGN_LOCK.json'),new_AB_C_experiments_executed=False,no_parameter_or_gate_reselection=True))
    put(BASE/'RESPONSE_CPU_COMPLETION.json',dict(status='complete',count=16,GT_read=True,result_sha256=sha(BASE/'RESPONSE_POSTSEAL_DIAGNOSIS.json')))
    print(json.dumps(dict(gate=gate)))
if __name__=='__main__':{'control_score':control_score,'response_root':response_root,'geometry':geometry}[sys.argv[1]]()
