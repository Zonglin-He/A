"""Per-group postseal evaluation; geometry and semantic identity stay separate."""
from pathlib import Path
import collections,gc,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE,PRIOR,read,put,sha,verify

def aggregate(rows):
    summary=[];paired=[]
    for ds in ('hc2','vidstg'):
        rr=[r for r in rows if r['dataset']==ds];ids=sorted({r['case_id']for r in rr});arms=list(dict.fromkeys(r['arm']for r in rr));idx={(r['case_id'],r['arm']):r['metrics']for r in rr};assert len(ids)==8 and len(idx)==8*len(arms)
        for a in arms:
            for key in idx[ids[0],a]:
                vv=[idx[c,a][key]for c in ids if idx[c,a][key]is not None];summary.append(dict(dataset=ds,arm=a,metric=key,valid=len(vv),unknown=8-len(vv),mean=math.fsum(vv)/len(vv)if vv else None))
        for a in arms:
            for b in arms:
                if a==b:continue
                for key in idx[ids[0],a]:
                    vv={c:100*(idx[c,a][key]-idx[c,b][key])for c in ids if idx[c,a][key]is not None and idx[c,b][key]is not None};v=list(vv.values());paired.append(dict(dataset=ds,comparison=a+'_minus_'+b,metric=key,valid=len(v),unknown=8-len(v),mean_delta_pp=math.fsum(v)/len(v)if v else None,positive=sum(x>1e-6 for x in v),negative=sum(x< -1e-6 for x in v),neutral=sum(abs(x)<=1e-6 for x in v),per_source_delta_pp=vv))
    return summary,paired

def semantic(arm,jobs):
    labels=read(BASE/'SEMANTIC_ANNOTATION_LOCK.json');assert labels['status']=='locked'and labels['annotator_blind_to_new_arm_choices'];idx={(j['case_id'],j['center_slot']):j for j in jobs};rows=[]
    for x in labels['records']:
        j=idx[x['case_id'],x['slot']]
        for name,p in [(arm,BASE/'teacher'/arm/(j['job_id']+'.json')),('Full_E1',PRIOR/'spatial/teacher'/(j['job_id']+'.json'))]:
            d=read(p);s=d['parsed'];visible=s.get('visible_id');valid=s['status']=='valid';category='invalid'if not valid else'abstain'if visible==0 else x['categories'].get(str(visible),'unknown');role='unknown'if not valid or visible==0 else x['role_status'].get(str(visible),'unknown')
            rows.append(dict(case_id=x['case_id'],dataset=j['dataset'],slot=x['slot'],arm=name,identity_known=x['identity_known'],selected_visible_id=visible,category=category,role_status=role,GT_read=False))
    sums=[]
    for ds in ('hc2','vidstg'):
        for a in (arm,'Full_E1'):
            r=[x for x in rows if x['dataset']==ds and x['arm']==a];known=[x for x in r if x['identity_known']];counts=dict(collections.Counter(x['category']for x in r));roles=dict(collections.Counter(x['role_status']for x in r));sums.append(dict(dataset=ds,arm=a,windows=len(r),identity_known_windows=len(known),category_counts=counts,same_entity_known=sum(x['category']in ('target_full','target_partial')for x in known),complete_target_known=sum(x['category']=='target_full'for x in known),unknown_or_mixed_known=sum(x['category']in ('unknown','mixed')for x in known),role_counts=roles,role_fixed_total=len(r)))
    return dict(rows=rows,summary=sums,annotation_sha256=sha(BASE/'SEMANTIC_ANNOTATION_LOCK.json'),GT_read=False,limits='single project-exposed annotator; original panels fixed before choices; sparse identity/role unknown retained; role-consistent selection is not proof of internal role reasoning')

def main(arm):
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.diagnose_stvg_lite_confirmation_spatial_v1 import iou
    verify();tick=time.perf_counter();b=BASE/'trials'/arm;root=read(b/'ROOT_READBACK.json');bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert root['status']=='pass'and root['count']==16 and bar['status']=='complete'and bar['count']==16 and not bar['GT_read']and bar['root_sha256']==sha(b/'ROOT_READBACK.json')
    bars={arm:bar,'Full_E1':read(PRIOR/'trials/E1/GLOBAL_PREDICTION_BARRIER.json')};maps={a:{f['case_id']:f for f in z['files']}for a,z in bars.items()};rr=read(BASE/'ROSTER_LOCK.json')['records'];ids={r['probe_alias']for r in rr};assert len(ids)==16
    for a in maps:
        for cid in ids:
            f=maps[a][cid];assert sha(ROOT/f['path'])==f['sha256']and sha(ROOT/f['receipt_path'])==f['receipt_sha256']
    # No GT import/access before both complete prediction streams and all SHAs pass.
    cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();old={(r['case_id'],r['arm']):r['metrics']for r in read(PRIOR/'EXPERIMENT_POSTSEAL_RESULTS.json')['rows']};refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];rows=[];usage=[];geometry=[];audits=[]
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];d=torch.load(ROOT/maps[arm][cid]['path'],map_location='cpu',weights_only=False);prev=torch.load(ROOT/maps['Full_E1'][cid]['path'],map_location='cpu',weights_only=False);pool=read(PRIOR/'spatial/E1/proposals'/f'{cid}.json');support=[w['source_grid_position']for w in pool['windows']];assert d['temporal_record']==prev['temporal_record']
        for name,pred in [('Frozen',d['arms']['F']),('Temporal',d['arms']['T']),('Full_E1',prev['arms']['TS_M2']),('Full_E1_Spatial',prev['arms']['S_M2']),(arm+'_Spatial',d['arms']['S_M2']),(arm,d['arms']['TS_M2'])]:
            scores,audit=metric._score_student_arm(pred,case,targets[cid],support,official,independent)
            if name in ('Frozen','Temporal','Full_E1'):assert scores==old[cid,'E1'if name=='Full_E1'else name],(cid,name)
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=scores,GT_read=True));audits.append(dict(case_id=cid,arm=name,audit=audit))
        f=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=rec['dataset'],anchors=len(refs[cid]['anchors']),actual_fit=f is not None,accepted_steps=f.get('optimizer_steps',0)if f else 0,nonzero_parameter_update=any(not torch.equal(v.cpu(),f['final_state'][k].cpu())for k,v in f['initial_state'].items())if f else False,fit_seconds=f.get('instance_study_fit_seconds')if f else None,decisions=refs[cid]['decisions'],GT_read=False))
        target=metric._truth_map(targets[cid]);scale=np.array([case['input']['width'],case['input']['height']]*2)
        for w in pool['windows']:
            fid=w['center_physical_frame_id'];g=(np.asarray(target[fid])/scale).tolist()if fid in target else None;j=next(j for j in jobs if j['case_id']==cid and j['center_slot']==w['slot']);outputs={a:read(p)['parsed']for a,p in [(arm,BASE/'teacher'/arm/(j['job_id']+'.json')),('Full_E1',PRIOR/'spatial/teacher'/(j['job_id']+'.json'))]};geometry.append(dict(case_id=cid,dataset=rec['dataset'],slot=w['slot'],known_GT=g is not None,candidates=[dict(candidate_id=t['candidate_id'],presence=t['presence'],GT_iou=iou(t['center_box'],g)if t['center_box']is not None and g is not None else None)for t in w['tracks']],selections={a:dict(parse_status=s['status'],selected_candidate_id=s.get('selected_candidate_id'))for a,s in outputs.items()},GT_read=True))
        del d,prev;gc.collect()
    summary,paired=aggregate(rows);result=dict(status='complete',count=16,arm=arm,rows=rows,summary=summary,paired=paired,usage=usage,geometry=geometry,semantic=semantic(arm,jobs),metric_audits=audits,GT_bindings=bindings,GT_read=True,prediction_barrier_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),scorer_sha256=sha(Path(__file__)),support='same four prospective E1 positions for all arms; common-observation support is not necessarily actual adopted training support',bootstrap=False)
    put(b/'POSTSEAL_RESULTS.json',result);put(b/'CPU_COMPLETION.json',dict(status='complete',count=16,row_count=len(rows),GT_read=True,result_sha256=sha(b/'POSTSEAL_RESULTS.json'),wall_seconds=time.perf_counter()-tick));print(json.dumps(dict(status='complete',arm=arm,summary=[r for r in summary if r['metric']=='m_vIoU'])))
if __name__=='__main__':main(sys.argv[1])
