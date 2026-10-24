"""Selected CG profile vs unchanged transferred profile on newly fixed CG parents."""
from common import *
import argparse,subprocess,time
from cg_calibrate import feedback

def prepare():
    p=validate('cg');freeze('cg_confirm',dict(selection_sha256=p['selection_sha256'],calibration_plan_sha256=sha(D/'cg_plan.json'),frozen_runtime_sha256={n:sha(R/'studies/paper_review_suite_v1'/n)for n in ['cgstvg_runtime_v9.py','cgstvg_interfaces_v9.py','cgstvg_memory_stream_v8.py','cgstvg_score.py']},selection_binding='After stageB selection, bind chosen parameter SHA before first confirmation model call. Wait actual TTA controller/child exit; native CG stays last.',arms=['SourceNative','Source','SharedCorrection','Space','Full','TransferredFull'],cohort='35HC+128Vid parents never evaluated by priorCG, fixed before new dev scores; all historical TA exposure disclosed',budget=dict(Source_max=167,same_ReferDINO8_max=163,wall_seconds=21600),precision='originalCG420FP32/nativegrid/crosssourceEMA',teacher_runtime_fingerprint=read(OLD/'budget_plan.json')['teacher_runtime_fingerprint'],old_results_immutable=True),['common.py','cg_calibrate.py','cg_confirm.py'])
    atomic(D/'cg_confirm_state.json',dict(status='prepared_waiting_development_and_TTA'))

def validate_all():
    p=validate('cg_confirm');validate('cg')
    for n,h in p['frozen_runtime_sha256'].items():assert sha(R/'studies/paper_review_suite_v1'/n)==h
    assert sha(D/'cg_selection.json')==p['selection_sha256'];return p

def exited(who):
    try:s=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split();return int(s[19])!=who['startticks']or s[0]=='Z'
    except FileNotFoundError:return True

def predict():
    import torch,numpy as np,gc
    from stvg.runtime import setup,lease
    from stvg.decode import decode
    from stvg.positions import observation_positions
    from stvg.teacher_bridge import ResidentTeacher
    from cgstvg_runtime_v9 import load_model,capture,feedback as original
    from cgstvg_interfaces_v9 import frame_grid
    p=validate_all();bind=read(D/'cg_confirm_binding.json');assert sha(D/'cg_B_selected.json')==bind['selected_sha256'];assert sha(D/'tta_confirm_score_receipt.json')==bind['tta_completion_sha256'];setup();gpu=lease();teacher=ResidentTeacher(cache_dir=D/'cg_teacher_cache',fingerprint=p['teacher_runtime_fingerprint']);files={};counts=dict(queries=0,Source_forwards=0,new_expert=0,native_bridges=0);started=time.perf_counter();atomic(D/'cg_confirm_worker.json',identity())
    try:
        for ds,groups in read(D/'cg_selection.json').items():
            tick=time.perf_counter();model,load=load_model(ds);load['model_load_seconds']=time.perf_counter()-tick;atomic(D/f'cg_confirm_{ds}_load.json',load);files[ds]={};trial=bind['selected'][ds]
            for i,row in enumerate(groups['confirm']):
                assert time.perf_counter()-started<p['budget']['wall_seconds'];key=row['id'];dest=D/'cg_confirm'/ds;dest.mkdir(parents=True,exist_ok=True);out=dest/(key+'.npz');cp=dest/(key+'.capture.pt');rp=dest/(key+'.receipt.json');assert not any(x.exists()for x in [out,cp,rp,dest/(key+'.acquired.pt')]),'Preserve paid acquisitions; never blind retry'
                tick=time.perf_counter();meta={**row['metadata'],'frame_ids':frame_grid(row['metadata'],ds)};frames,ids=decode(meta,ds);assert ids==meta['frame_ids'];atomic(dest/(key+'.attempt.json'),dict(worker=identity(),phase='before_Source',GT_read=False));cap=capture(model,frames,ids,meta['caption'],noop=i<2);counts['Source_forwards']+=cap['source_forwards'];counts['native_bridges']+=int(cap['native_noop']);iv=cap['native_map'];pos=observation_positions(ids,[ids.index(iv[0]),ids.index(iv[1]-1)]);cap['positions8']=pos;native=teacher(frames[pos],meta['caption'],positions=pos,frame_ids=ids,dataset=ds);native['positions']=pos;counts['new_expert']+=native['query_forwards'];ap=dest/(key+'.acquired.pt');torch.save(dict(Source_capture=cap,teacher=native),ap);atomic(dest/(key+'.acquired.json'),dict(sha256=sha(ap),source_forwards=cap['source_forwards'],new_expert=native['query_forwards'],GT_read=False));arrays=feedback(cap,native,ds,trial);base,details,_=original(cap,native,ds);arrays.update(TransferredFull=base['Full'],TransferredFull_interval=base['Full_interval'])
                if i<2:
                    replay=feedback(cap,native,ds,trial);assert all(np.array_equal(arrays[k],replay[k])for k in replay)
                np.savez_compressed(out,**arrays);torch.save(cap,cp);rec=dict(path=str(out.relative_to(D)),sha256=sha(out),capture_path=str(cp.relative_to(D)),capture_sha256=sha(cp),acquired_sha256=sha(ap),source_forwards=cap['source_forwards'],Source_seconds=cap['source_seconds'],expert_new=native['query_forwards'],expert_cost=native.get('evidence_original_cost',native['cost']),model_load=load,teacher_load_seconds=native['model_load_seconds'],pipeline_seconds=time.perf_counter()-tick,trial=trial,worker=identity(),GT_read=False);atomic(rp,rec);files[ds][key]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)};counts['queries']+=1;atomic(D/'cg_confirm_progress.json',dict(dataset=ds,domain_completed=i+1,counts=counts,seconds=time.perf_counter()-started,GT_read=False,**identity()));print(ds,i+1,counts,flush=True);del frames,cap,native,arrays,base;gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
        assert counts['queries']==163 and counts['native_bridges']==4 and counts['Source_forwards']==167 and counts['new_expert']<=163;atomic(D/'cg_confirm_sealed.json',dict(files=files,counts=counts,GT_read=False,fingerprint=p['fingerprint'],binding_sha256=sha(D/'cg_confirm_binding.json'),seconds=time.perf_counter()-started))
    finally:teacher.close();gpu.close()

def score():
    import torch,numpy as np
    from full_score import metric
    from cgstvg_score import native_official_metric
    p=validate_all();torch.set_num_threads(1);seal=read(D/'cg_confirm_sealed.json');assert seal['counts']['queries']==163 and not seal['GT_read'];checks=0;maxerr=0.
    for ds,groups in read(D/'cg_selection.json').items():
        truth=labels(ds);rows=[]
        for row in groups['confirm']:
            g=truth[row['id']];gt={int(k):v for k,v in g['boxes'].items()};e=seal['files'][ds][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['capture_path'])==e['capture_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];vals={}
            with np.load(D/e['path'])as z:
                r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
                for arm in p['arms']:
                    iv=z[arm+'_interval'].tolist();vals[arm],err=metric(z[arm],r,gt,g['interval'],iv,z['positions8']);ref=native_official_metric(z[arm],z['frame_ids'].tolist(),r['metadata'],gt,g['interval'],iv,ds);err=max(err,max(abs(vals[arm][k]-ref[k])for k in ['v','t','s']));assert err<2e-10;maxerr=max(maxerr,err);checks+=1
            rows.append(dict(id=row['id'],parent=str(g['parent']),values=vals,duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s'))
        res=report(rows,p['arms'],[('Full','Source'),('Full','TransferredFull'),('Full','Space')],P/'cg_confirm'/ds);atomic(D/'cg_confirm'/ds/'metrics.json',dict(report=res,private_rows=rows));atomic(P/'cg_confirm'/ds/'cost.json',dict(receipts=list(seal['files'][ds].values()),cache_is_not_deployment_speedup=True))
    atomic(D/'cg_confirm_score_receipt.json',dict(checks=checks,max_error=maxerr,seal_sha256=sha(D/'cg_confirm_sealed.json'),all163sealed_before_GT=True))

def run():
    p=validate_all();atomic(D/'cg_confirm_launch.json',identity())
    while not all((D/n).exists()for n in ['cg_B_selected.json','cg_B_score_receipt.json','tta_confirm_score_receipt.json']):
        for name in ['cg_controller_launch.json','tta_controller_launch.json']:
            if(D/name).exists()and exited(read(D/name)):
                necessary='cg_B_score_receipt.json'if name.startswith('cg_')else'tta_confirm_score_receipt.json'
                assert(D/necessary).exists(),'Dependency exited without completion; preserve and repair'
        time.sleep(10)
    while any(not exited(read(D/n))for n in ['cg_controller_launch.json','tta_controller_launch.json']):time.sleep(5)
    assert not(D/'cg_confirm_binding.json').exists();atomic(D/'cg_confirm_binding.json',dict(selected=read(D/'cg_B_selected.json'),selected_sha256=sha(D/'cg_B_selected.json'),tta_completion_sha256=sha(D/'tta_confirm_score_receipt.json'),selection_sha256=p['selection_sha256'],GT_confirmation_not_read=True))
    for command in ['predict','score']:
        atomic(D/'cg_confirm_state.json',dict(status=command,**identity(),fingerprint=p['fingerprint']));child=subprocess.Popen([sys.executable,'-B',__file__,command]);atomic(D/f'cg_confirm_{command}_child.json',identity(child.pid));assert child.wait()==0,'Preserve acquisition/failure; never blind retry'
    atomic(D/'cg_confirm_state.json',dict(status='complete_scored_pending_publication',**identity(),fingerprint=p['fingerprint']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','predict','score','run']);z=a.parse_args();{'prepare':prepare,'predict':predict,'score':score,'run':run}[z.command]()
