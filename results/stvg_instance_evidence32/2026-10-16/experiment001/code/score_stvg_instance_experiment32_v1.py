"""Score all matched Student arms only after joint E0/E1 prediction sealing."""
from pathlib import Path
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
from scripts.score_stvg_instance32_v1 import aggregate
def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.diagnose_stvg_lite_confirmation_spatial_v1 import iou
    bar=read(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==32 and not bar['GT_read']
    gate=read(BASE/'ROOT_LOCKED_FIT_BINDING.json');assert gate['status']=='pass'and gate['prediction_barrier_sha256']==sha(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json')and gate['scorer_sha256']==sha(Path(__file__))
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};targets,gt=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions()
    filemaps={};refs={}
    for tid in ('E0','E1'):
        b=BASE/'trials'/tid;assert sha(b/'GLOBAL_PREDICTION_BARRIER.json')==bar['trials'][tid]and read(b/'ROOT_READBACK.json')['status']=='pass'
        filemaps[tid]={f['case_id']:f for f in read(b/'STUDENT_GLOBAL_BARRIER.json')['files']};refs[tid]={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']}
    rows=[];geometry=[];usage=[]
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];dd={}
        for tid in ('E0','E1'):
            f=filemaps[tid][cid];assert sha(ROOT/f['path'])==f['sha256'];dd[tid]=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
        assert dd['E0']['temporal_record']==dd['E1']['temporal_record']and dd['E0']['arms']['T']['interval']==dd['E1']['arms']['T']['interval']
        support=[w['source_grid_position']for w in read(BASE/'spatial/E0/proposals'/f'{cid}.json')['windows']]
        arms={'Frozen':dd['E0']['arms']['F'],'Temporal':dd['E0']['arms']['T'],'E0':dd['E0']['arms']['TS_M2'],'E1':dd['E1']['arms']['TS_M2'],'E2':dd['E1']['arms']['TS_M0']}
        for name,pred in arms.items():
            scores,audit=metric._score_student_arm(pred,case,targets[cid],support,official,independent)
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=scores,common_observation_count=len(support),GT_read=True))
        for tid,mode,arm in [('E0','M2','E0'),('E1','M2','E1'),('E1','M0','E2')]:
            d=dd[tid];f=d['fits'][mode]
            nonzero=False
            if f is not None:
                nonzero=any(not torch.equal(v.cpu(),f['final_state'][k].cpu())for k,v in f['initial_state'].items())
            usage.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,anchors=len(refs[tid][cid]['anchors']),actual_fit=f is not None,accepted_steps=f.get('optimizer_steps',0)if f else 0,nonzero_parameter_update=nonzero,fit_seconds=f.get('instance_study_fit_seconds',f.get('timing',{}))if f else None,decisions=refs[tid][cid]['decisions'],GT_read=False))
        target=metric._truth_map(targets[cid]);scale=np.array([case['input']['width'],case['input']['height']]*2)
        for tid in ('E0','E1'):
            pool=read(BASE/'spatial'/tid/'proposals'/f'{cid}.json')
            for w in pool['windows']:
                fid=w['center_physical_frame_id'];gtbox=(np.asarray(target[fid])/scale).tolist()if fid in target else None
                decision=next(x for x in refs[tid][cid]['decisions']if x['slot']==w['slot']);candidates=[]
                for t in w['tracks']:
                    value=iou(t['center_box'],gtbox)if t['center_box']is not None and gtbox is not None else None
                    srcbox=w['tracks'][0]['center_box'];same=t['center_box']==srcbox
                    source_iou=iou(t['center_box'],srcbox)if t['center_box']is not None and srcbox is not None else None
                    candidates.append(dict(candidate_id=t['candidate_id'],role='Source'if t['candidate_id']==1 else'proposal',selected=t['candidate_id']==decision['selected_candidate_id'],presence=t['presence'],GT_iou=value,geometry_exactly_Source=same,IoU_with_Source=source_iou))
                geometry.append(dict(case_id=cid,dataset=rec['dataset'],group=tid,slot=w['slot'],known_GT=gtbox is not None,candidates=candidates,decision=decision,GT_read=True))
        del dd;gc.collect()
    summary,paired=aggregate(rows)
    put(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,usage=usage,geometry=geometry,GT_bindings=gt,GT_read=True,prediction_barrier_sha256=sha(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json'),E2='E1 reference lock/M0',support='same four prospective observation positions for all five arms; remaining source-grid GT positions form unsupervised readout'))
    put(BASE/'EXPERIMENT_CPU_COMPLETION.json',dict(status='complete',count=32,row_count=len(rows),GT_read=True,result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json')))
    print(json.dumps(dict(status='complete',paired=[r for r in paired if r['metric']=='m_vIoU'])))
if __name__=='__main__':main()
