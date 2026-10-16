"""Postseal extent and guarded-update effects; no GT-dependent decisions."""
from pathlib import Path
import gc,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_B_transfer_v1 import BASE,PRIOR,read,put,sha,check
from scripts.score_stvg_referent_abc_v1 import aggregate
def geom(box,truth):
    import numpy as np
    c=np.asarray(box,float);xy=np.r_[c[:2]-c[2:]/2,c[:2]+c[2:]/2];g=np.asarray(truth,float);inter=np.maximum(0,np.minimum(xy[2:],g[2:])-np.maximum(xy[:2],g[:2])).prod();pa=np.maximum(0,xy[2:]-xy[:2]).prod();ga=np.maximum(0,g[2:]-g[:2]).prod();return dict(IoU=float(inter/max(pa+ga-inter,1e-12)),target_coverage=float(inter/max(ga,1e-12)),reference_precision=float(inter/max(pa,1e-12)),reference_area=float(pa))
def main(arm):
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    check(arm);tick=time.perf_counter();b=BASE/'trials'/arm;bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==16 and not bar['GT_read']and bar['root_sha256']==sha(b/'ROOT_READBACK.json');assert read(b/'ROOT_READBACK.json')['status']=='pass';old=PRIOR/'trials/B';bars={arm:bar,'B00':read(old/'GLOBAL_PREDICTION_BARRIER.json')};maps={a:{f['case_id']:f for f in x['files']}for a,x in bars.items()};rr=read(BASE/'ROSTER_LOCK.json')['records']
    for rec in rr:
        for a in maps:
            f=maps[a][rec['probe_alias']];assert sha(ROOT/f['path'])==f['sha256']and sha(ROOT/f['receipt_path'])==f['receipt_sha256']
    cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();oldscore={(r['case_id'],r['arm']):r['metrics']for r in read(old/'POSTSEAL_RESULTS.json')['rows']};refs={a:{r['case_id']:r for r in read(p/'REFERENCE_LOCK.json')['records']}for a,p in [(arm,b),('B00',old)]};rows=[];actual_support=[];geometry=[];usage=[];audits=[]
    for rec in rr:
        cid=rec['probe_alias'];c=cases[cid];d={a:torch.load(ROOT/maps[a][cid]['path'],map_location='cpu',weights_only=False)for a in maps};assert d[arm]['temporal_record']==d['B00']['temporal_record'];positions={a:[x['position']for x in refs[a][cid]['anchors']]for a in maps};assert positions[arm]==positions['B00'];common=[w['source_grid_position']for w in read(ROOT/'artifacts/stvg_instance_evidence32_v1/spatial/E1/proposals'/f'{cid}.json')['windows']]
        preds={'Frozen':d[arm]['arms']['F'],'Temporal':d[arm]['arms']['T'],'B00':d['B00']['arms']['TS_M2'],'B00_Spatial':d['B00']['arms']['S_M2'],arm:d[arm]['arms']['TS_M2'],arm+'_Spatial':d[arm]['arms']['S_M2']}
        for name,pred in preds.items():
            scores,audit=metric._score_student_arm(pred,c,targets[cid],common,official,independent)
            if name in ('Frozen','Temporal','B00'):assert scores==oldscore[cid,'B'if name=='B00'else name]
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=scores,GT_read=True));audits.append(dict(case_id=cid,arm=name,common=audit))
            s,a=metric._score_student_arm(pred,c,targets[cid],positions['B00'],official,independent);actual_support.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=s,adopted_anchor_count=len(positions['B00']),known_supervised=a['support']['supervised_gt_known_frame_count'],known_unsupervised=a['support']['unsupervised_gt_known_frame_count'],GT_read=True))
        for name in maps:
            fit=d[name]['fits']['M2'];steps=[]
            if fit is not None:
                for row in fit['path']:
                    before=row.get('reference_iou_before');cc=[]
                    for z in row.get('candidates',[]):
                        cc.append(dict(scale=z['scale'],finite=z['finite'],reference_pass=z['reference_pass'],accepted=z['accepted'],agreement_gain=float(z['anchor_IoU'].double().mean()-before.double().mean())))
                    steps.append(dict(step=row['step'],accepted=row['accepted'],accepted_scale=row['accepted_scale'],skipped=row['skipped'],reason=row.get('reason'),rollback=row['rollback'],candidates=cc))
            usage.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,anchors=len(positions[name]),actual_fit=fit is not None,accepted_steps=fit['optimizer_steps']if fit else 0,nonzero_final=any(not torch.equal(v.cpu(),fit['final_state'][k].cpu())for k,v in fit['initial_state'].items())if fit else False,fit_seconds=fit.get('instance_study_fit_seconds')if fit else None,steps=steps,GT_read=False))
        tm=metric._truth_map(targets[cid]);scale=np.array([c['input']['width'],c['input']['height']]*2)
        for orig,new in zip(refs['B00'][cid]['anchors'],refs[arm][cid]['anchors']):
            assert {k:v for k,v in orig.items()if k!='box_cxcywh'}=={k:v for k,v in new.items()if k!='box_cxcywh'};fid=orig['physical_frame_id'];g=np.asarray(tm[fid])/scale if fid in tm else None;geometry.append(dict(case_id=cid,dataset=rec['dataset'],anchor_ordinal=len([x for x in geometry if x['case_id']==cid]),candidate_id=orig['candidate_id'],known_GT=g is not None,B00=geom(orig['box_cxcywh'],g)if g is not None else None,intervention=geom(new['box_cxcywh'],g)if g is not None else None,center_exact=orig['box_cxcywh'][:2]==new['box_cxcywh'][:2],width_ratio=new['box_cxcywh'][2]/orig['box_cxcywh'][2],height_ratio=new['box_cxcywh'][3]/orig['box_cxcywh'][3],GT_read=True))
        del d;gc.collect()
    summary,paired=aggregate(rows);sactual,pactual=aggregate(actual_support);result=dict(status='complete',count=16,arm=arm,rows=rows,summary=summary,paired=paired,actual_support_rows=actual_support,actual_support_summary=sactual,actual_support_paired=pactual,geometry=geometry,usage=usage,metric_audits=audits,GT_bindings=bindings,GT_read=True,scorer_sha256=sha(Path(__file__)),prediction_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),geometry_limit='same-identity center-preserving extent perturbation, not certified full object; GT only postseal; no geometry winner used for training')
    put(b/'POSTSEAL_RESULTS.json',result);put(b/'CPU_COMPLETION.json',dict(status='complete',count=16,row_count=len(rows),GT_read=True,result_sha256=sha(b/'POSTSEAL_RESULTS.json'),wall_seconds=time.perf_counter()-tick));print(json.dumps(dict(status='complete',arm=arm,contrasts=[r for r in paired if r['metric']=='m_vIoU'and r['comparison']in (arm+'_minus_B00',arm+'_minus_Temporal')])) )
if __name__=='__main__':main(sys.argv[1])
