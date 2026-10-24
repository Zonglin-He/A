"""Independent selected CPU full validation, only for replayable output families."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import time
import fcntl
import shutil
import numpy as np
import torch
from stvg.runtime import ROOT, sha
from stvg.tube_runner import write,digest
from .run import PUBLIC,OUT,PLAN,checked,verify_seal
from .numeric import finish

FULL_PLAN=PUBLIC/'full_plan.json'


def cpu_replayable(p):
    return p['family'] in ['source_control','source_calibration','sealed_full_blend','initialization'] or (p['alpha_observed']==0 and p['alpha_unobserved']==0)


def prepare():
    assert not FULL_PLAN.exists()
    p=checked();lock_path=PUBLIC/'selected_profiles.json';locks=json.loads(lock_path.read_text())
    assert locks['status']=='both128_locked_before_new_full' and locks['plan_sha256']==sha(PLAN)
    can={ds:cpu_replayable(s['winner']['parameters']) for ds,s in locks['datasets'].items()}
    full=dict(status='frozen_before_selected_complete_populations',search_plan_sha256=sha(PLAN),
       locks_sha256=sha(lock_path),locks=locks['datasets'],CPU_replayable=can,
       policy='no new selection from full; keep unchanged Source, original native time and prior Selected_interval; labels only after complete verified new prediction seal',
       current_LLaVA_GPU_not_interrupted=True,code_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'tools/tubedetr_spatial_search/numeric.py']})
    parent=json.loads((ROOT/'results/simplepath_tubedetr/run_plan.json').read_text())
    full['native_dependency_sha256']=parent['code_sha256']
    for name,h in full['native_dependency_sha256'].items():assert sha(ROOT/name)==h,name
    write(FULL_PLAN,full);print(json.dumps(can),flush=True)


def full_checked():
    p=checked();f=json.loads(FULL_PLAN.read_text());assert f['search_plan_sha256']==sha(PLAN)
    assert f['locks_sha256']==sha(PUBLIC/'selected_profiles.json')
    for name,h in f['code_sha256'].items():assert sha(ROOT/name)==h
    for name,h in f['native_dependency_sha256'].items():assert sha(ROOT/name)==h,name
    return p,f


def payload(path,expected):
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def predict(ds):
    from stvg.runtime import setup
    setup();p,f=full_checked();assert f['CPU_replayable'][ds], 'Selected geometry needs real native feature GPU after LLaVA'
    pars=f['locks'][ds]['winner']['parameters'];d=p['datasets'][ds]
    root=(ROOT/d['fixed_full_seal']).parent;s=json.loads((root/'sealed.json').read_text())
    assert sha(root/'sealed.json')==d['fixed_full_seal_sha256'] and s['status']=='complete'
    assert sha(ROOT/d['full_manifest'])==d['full_manifest_sha256']
    rows=list(map(json.loads,(ROOT/d['full_manifest']).read_text().splitlines()));assert len(rows)==d['full_queries'] and set(s['files'])=={r['id'] for r in rows}
    # Original consensus deltas are retained in the pre-existing Source receipts.
    oldplan=json.loads((ROOT/'results/tubedetr_boxfixed_development/plan.json').read_text())
    native_root=ROOT/oldplan['datasets'][ds]['old_reference'];native_seal=json.loads((native_root/'sealed.json').read_text())
    assert sha(native_root/'sealed.json')==oldplan['datasets'][ds]['old_prediction_seal_sha256']
    target=OUT/ds/'full';target.mkdir(parents=True,exist_ok=True);entries={}
    for i,row in enumerate(rows):
        if shutil.disk_usage(ROOT).free < (25<<30)+(64<<20):raise RuntimeError('25GiB free-space floor; all evidence retained')
        q=row['id'];e=s['files'][q];a=payload(root/e['path'],e['sha256'])
        assert sha(root/e['receipt_path'])==e['receipt_sha256']
        obs=np.zeros(len(a['Source']),bool);obs[a['positions8']]=True
        source=np.asarray(a['Source'],np.float64)
        if pars['family']=='sealed_full_blend':pred=a['Full']
        elif pars['family']=='initialization':
            from stvg.consensus import apply
            from stvg.predictor import output_writeback
            ne=native_seal['files'][q];rp=native_root/ne['receipt_path'];assert sha(rp)==ne['receipt_sha256']
            receipt=json.loads(rp.read_text());delta=np.asarray(receipt['initialization']['delta'])
            raw=apply(source,delta);pred=output_writeback(source,raw,pars['gamma'])
        else:pred=source
        b=finish(source,pred,obs,pars)
        path=target/(q+'.npz');tmp=path.with_suffix('.pending')
        with tmp.open('xb') as stream:np.savez_compressed(stream,Source=a['Source'],Ours=b,frame_ids=a['frame_ids'],positions8=a['positions8'],Native_interval=a['Native_interval'],Selected_interval=a['Selected_interval'])
        tmp.replace(path);rp=path.with_suffix('.json')
        write(rp,dict(search_plan_sha256=sha(PLAN),full_plan_sha256=sha(FULL_PLAN),request_sha256=digest(row),payload_sha256=sha(path),
             labels_read=False,Source_unchanged=True,parameters=pars,old_fixed_payload_sha256=e['sha256'],new_neural_calls=0,new_teacher_calls=0))
        entries[q]=dict(path=path.name,sha256=sha(path),receipt_path=rp.name,receipt_sha256=sha(rp))
        if (i+1)%100==0 or i==len(rows)-1:
            write(OUT/'worker_progress.json',dict(dataset=ds,stage='selected_full_CPU',complete=i+1,total=len(rows),labels_read=False));print(ds,i+1,'/',len(rows),flush=True)
    write(target/'sealed.json',dict(status='complete',queries=len(rows),files=entries,search_plan_sha256=sha(PLAN),full_plan_sha256=sha(FULL_PLAN),labels_read=False,Source_unchanged=True))


def native_capture(model,row,p,ds,catalog):
    from stvg.tube_decode import decode
    from tools.boxfixed_tube.adapter import capture
    from tools.boxfixed_tube.evidence import SourceEvidence
    from stvg.initialization import initialize
    from stvg.positions import observation_positions
    frames,ids=decode(row['metadata'],ds)
    a=capture(model,frames,ids,row['metadata']['caption'])
    positions=observation_positions(ids,a['native_extent'])
    parent=json.loads((ROOT/'results/tubedetr_boxfixed_development/plan.json').read_text())
    teacher,trace=catalog.teacher('tubedetr',frames[positions],row['metadata']['caption'],parent['teacher_fingerprint'])
    evidence=SourceEvidence.from_native_masks(a['features'],a['valid'],ids,positions,teacher['mask_logits'],teacher['teacher_image_size'],a['grid_shape'],selected_query=teacher['selected_query'])
    source=a['Source'].numpy()
    raw,init=initialize(source.astype(np.float64),positions,teacher['boxes'].double().numpy(),p['datasets'][ds]['configuration']['E_calibration'])
    return dict(Source=source,raw=raw,Native_interval=a['Native_interval'],frame_ids=ids,positions8=positions,evidence=evidence,native_boxes=teacher['boxes'].double()),trace


def selected_prediction(c,a,p,f,ds):
    from .numeric import predictions
    pars=f['locks'][ds]['winner']['parameters']
    profile=[p['profiles'][ds][0],p['profiles'][ds][1],pars]
    return predictions(c,a,profile,p['geometry_variants'])[2]


def gpu(ds):
    """Only after LLaVA completion; qualification and full label-free inference."""
    from stvg.runtime import setup
    from stvg.tubedetr import load_source
    from stvg.tensors import state_hash
    from tools.reuse_cache.catalog import Catalog
    setup();torch.set_num_threads(1);p,f=full_checked();d=p['datasets'][ds]
    assert json.loads((ROOT/'outputs/simplepath_llava/status.json').read_text())['status']=='complete_both_fixed128_and_full'
    rows=list(map(json.loads,(ROOT/d['full_manifest']).read_text().splitlines()))
    fixedroot=(ROOT/d['fixed_full_seal']).parent;fixed=json.loads((fixedroot/'sealed.json').read_text())
    assert sha(fixedroot/'sealed.json')==d['fixed_full_seal_sha256']
    dev=OUT/ds/'development';devseal=verify_seal(dev/'sealed.json');index=f['locks'][ds]['winner']['index']
    cat=Catalog(ROOT/'runtime/reuse_cache/catalog.sqlite');pars=f['locks'][ds]['winner']['parameters']
    target=OUT/ds/'full';target.mkdir(parents=True,exist_ok=True);entries={};calls=0
    with (ROOT/'runtime/gpu.lock').open('a') as lock:
        print('Waiting for shared gpu.lock after LLaVA completion',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        model=load_source(d['configuration']);before=state_hash(model.state_dict())
        # One original development query proves actual feature and selected-output replay.
        row=json.loads((ROOT/d['manifest']).read_text().splitlines()[0]);q=row['id']
        c,trace=native_capture(model,row,p,ds,cat);calls+=1
        caproot=(ROOT/d['captures_seal']).parent;caps=json.loads((caproot/'sealed.json').read_text());ce=caps['files'][q]
        assert sha(caproot/ce['path'])==ce['sha256'];reference=torch.load(caproot/ce['path'],map_location='cpu',weights_only=False)
        for key in ['Source','raw','Native_interval','frame_ids','positions8']:assert np.array_equal(np.asarray(c[key]),np.asarray(reference[key])),('GPU replay',ds,key)
        assert torch.equal(c['evidence'].features,reference['evidence'].features) and torch.equal(c['evidence'].seeds,reference['evidence'].seeds)
        a=payload(fixedroot/fixed['files'][q]['path'],fixed['files'][q]['sha256'])
        b=selected_prediction(c,a,p,f,ds)
        with np.load(dev/devseal['files'][q]['path'],allow_pickle=False) as z:assert np.array_equal(b,z['predictions'][index])
        assert before==state_hash(model.state_dict())
        write(PUBLIC/ds/'selected_GPU_qualification.json',dict(status='pass',existing_query=True,features_Source_raw_selected_output_bitexact=True,new_neural_calls=1,new_teacher_calls=0,labels_read=False,full_plan_sha256=sha(FULL_PLAN)))
        for i,row in enumerate(rows):
            if shutil.disk_usage(ROOT).free < (25<<30)+(64<<20):raise RuntimeError('25GiB free-space floor; all evidence retained')
            q=row['id'];e=fixed['files'][q];a=payload(fixedroot/e['path'],e['sha256'])
            assert sha(fixedroot/e['receipt_path'])==e['receipt_sha256']
            if q in devseal['files']:
                de=devseal['files'][q];rp=dev/de['receipt_path'];assert sha(rp)==de['receipt_sha256'] and json.loads(rp.read_text())['request_sha256']==digest(row)
                da=payload(dev/de['path'],de['sha256']);b=da['predictions'][index];trace=dict(reused_selected128=True)
            else:
                c,trace=native_capture(model,row,p,ds,cat);calls+=1
                for key in ['Source','Native_interval','frame_ids','positions8']:assert np.array_equal(np.asarray(c[key]),a[key]),('old full Source drift',ds,q,key)
                b=selected_prediction(c,a,p,f,ds)
            path=target/(q+'.npz');tmp=path.with_suffix('.pending')
            with tmp.open('xb') as stream:np.savez_compressed(stream,Source=a['Source'],Ours=b,frame_ids=a['frame_ids'],positions8=a['positions8'],Native_interval=a['Native_interval'],Selected_interval=a['Selected_interval'])
            tmp.replace(path);rp=path.with_suffix('.json')
            write(rp,dict(search_plan_sha256=sha(PLAN),full_plan_sha256=sha(FULL_PLAN),request_sha256=digest(row),payload_sha256=sha(path),parameters=pars,
                          labels_read=False,Source_unchanged=True,new_teacher_calls=0,teacher_trace=trace))
            entries[q]=dict(path=path.name,sha256=sha(path),receipt_path=rp.name,receipt_sha256=sha(rp))
            write(OUT/'worker_progress.json',dict(dataset=ds,stage='selected_full_GPU',complete=i+1,total=len(rows),labels_read=False));print(ds,i+1,'/',len(rows),flush=True)
        assert before==state_hash(model.state_dict()) and all(not x.requires_grad and x.grad is None for x in model.parameters())
        write(target/'sealed.json',dict(status='complete',queries=len(rows),files=entries,search_plan_sha256=sha(PLAN),full_plan_sha256=sha(FULL_PLAN),labels_read=False,Source_unchanged=True,model_state_unchanged=True,new_teacher_calls=0,new_neural_calls=calls))
    cat.db.close()


def score(ds):
    p,f=full_checked();d=p['datasets'][ds];target=OUT/ds/'full';s=verify_seal(target/'sealed.json')
    assert s['queries']==d['full_queries'] and s['full_plan_sha256']==sha(FULL_PLAN)
    assert sha(ROOT/d['full_labels'])==d['full_labels_sha256']
    truth={r['id']:r for r in map(json.loads,(ROOT/d['full_labels']).read_text().splitlines())}
    rows=list(map(json.loads,(ROOT/d['full_manifest']).read_text().splitlines()));assert set(truth)==set(s['files'])=={r['id'] for r in rows}
    details=[]
    from stvg.metrics import DenseTube
    for row in rows:
        q=row['id'];gt=truth[q];e=s['files'][q];values={};unobserved={}
        with np.load(target/e['path'],allow_pickle=False) as a:
            seen=set(a['frame_ids'][a['positions8']].tolist());meta=dict(frame_ids=a['frame_ids'],input=row['metadata'])
            for arm,iv in [('Source','Native_interval'),('Ours','Selected_interval')]:
                m=DenseTube(a[arm],meta,{int(k):v for k,v in gt['boxes'].items()},gt['interval'],clip=ds=='hc2');values[arm]=m.score(a[iv]);unobserved[arm]=float(m.iou[[int(t) not in seen for t in m.fids]].mean())
        details.append(dict(id=q,parent=gt['parent'],values=values,unobserved=unobserved))
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    result=dict(status='complete',dataset=ds,queries=len(rows),parents=len({x['parent'] for x in details}),
      means={arm:{k:float(np.mean([x['values'][arm][k] for x in details])*100) for k in ['s','v','t']} for arm in ['Source','Ours']},
      unobserved_sIoU={arm:float(np.mean([x['unobserved'][arm] for x in details])*100) for arm in ['Source','Ours']},
      paired={k:cluster_difference([x['values']['Ours'][k]-x['values']['Source'][k] for x in details],[x['parent'] for x in details]) for k in ['s','v']},
      prediction_seal_sha256=sha(target/'sealed.json'),search_plan_sha256=sha(PLAN),full_plan_sha256=sha(FULL_PLAN),parameters=f['locks'][ds]['winner']['parameters'],
      full_used_for_selection=False,previous_research_exposure=True,independent_unexposed_test=False)
    result['delta_sIoU_pp']=result['means']['Ours']['s']-result['means']['Source']['s']
    write(PUBLIC/ds/'full_metrics.json',result)
    with gzip.open(PUBLIC/ds/'all_signed_full_results.json.gz','wt') as stream:
        json.dump([dict(query=hashlib.sha256(x['id'].encode()).hexdigest(),parent=hashlib.sha256(str(x['parent']).encode()).hexdigest(),values=x['values'],unobserved=x['unobserved']) for x in details],stream,separators=(',',':'))
    write(PUBLIC/ds/'final_seal.json',dict(status='complete',files={str(z.relative_to(ROOT)):sha(z) for z in [FULL_PLAN,PUBLIC/ds/'selected_profile.json',PUBLIC/ds/'full_metrics.json',PUBLIC/ds/'all_signed_full_results.json.gz']},prediction_seal_sha256=sha(target/'sealed.json'),publication_pending=True))
    print(ds,json.dumps(result['means']),result['delta_sIoU_pp'],flush=True)


def run():
    p,f=full_checked()
    for ds in p['dataset_order']:
        if not f['CPU_replayable'][ds]:
            while True:
                if (ROOT/'outputs/simplepath_llava/failure.json').exists():raise RuntimeError('LLaVA has unresolved failure; do not bypass task ordering')
                status=json.loads((ROOT/'outputs/simplepath_llava/status.json').read_text())
                if status.get('status')=='complete_both_fixed128_and_full':break
                write(OUT/'full_status.json',dict(status='waiting_for_LLaVA_before_native_geometry_GPU',dataset=ds,selection_locked=True));time.sleep(30)
        for stage in [('predict' if f['CPU_replayable'][ds] else 'gpu'),'score']:
            log=ROOT/'logs'/f'tubedetr_spatial_search_{ds}_full_{stage}.log'
            cmd=[sys.executable,'-B','-u','-m','tools.tubedetr_spatial_search.full',stage,'--dataset',ds]
            with log.open('a') as stream:
                env=dict(os.environ)
                if stage!='gpu':env['CUDA_VISIBLE_DEVICES']=''
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
                write(ROOT/'runtime/tubedetr_spatial_search_full.json',dict(controller_pid=os.getpid(),worker_pid=proc.pid,command=cmd,log=str(log.relative_to(ROOT)),dataset=ds,stage=stage))
                if proc.wait():raise RuntimeError(f'{ds} full {stage} failed')
    files=[PLAN,FULL_PLAN,PUBLIC/'selected_profiles.json',*[PUBLIC/ds/'final_seal.json' for ds in p['dataset_order']]]
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(z.relative_to(ROOT)):sha(z) for z in files},publication_pending=True))
    write(OUT/'full_status.json',dict(status='both_complete_populations_scored',Root_verification_and_publication_pending=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run','predict','gpu','score']);a.add_argument('--dataset',choices=['vidstg','hc2']);args=a.parse_args()
    try:globals()[args.stage]() if args.stage in ['prepare','run'] else globals()[args.stage](args.dataset)
    except Exception as exc:
        write(OUT/'full_failure.json',dict(stage=args.stage,dataset=args.dataset,error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
