"""GitHub BoxFixed/Global6 TubeDETR: sealed128 -> bounded tuning -> selected full.

GPU workers never read labels. Each CPU evaluation requires complete evidence
and prediction seals. Profiles share native evidence; no optimizer or448 branch.
"""
import argparse,copy,hashlib,json,os,shlex,subprocess,sys,time,fcntl,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup
from stvg.tube_runner import write,digest
from tools.boxfixed_tube.profiles import baseline,profile_id,predict
PUBLIC=ROOT/'results/tubedetr_boxfixed_development';OUT=ROOT/'outputs/tubedetr_boxfixed_development';PLAN=PUBLIC/'plan.json'


def checked():
    p=json.loads(PLAN.read_text());assert sha(ROOT/'configs/tubedetr_boxfixed.json')==p['configuration_sha256']
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    for ds,d in p['datasets'].items():
        for name in ['manifest','full_manifest']:
            assert sha(ROOT/d[name])==d[name+'_sha256'],(ds,name)
    return p


def records(ds,full=False):
    p=checked();d=p['datasets'][ds];path=ROOT/d['full_manifest'if full else'manifest'];rows=[json.loads(x)for x in path.read_text().splitlines()];assert len(rows)==(d['full_queries']if full else 128)
    assert len({r['id']for r in rows})==len(rows);return p,rows


def old_reference(ds,p):
    root=ROOT/p['datasets'][ds]['old_reference'];f=root/'sealed.json';assert sha(f)==p['datasets'][ds]['old_prediction_seal_sha256'];return root,json.loads(f.read_text())


def verify_old(root,seal,row,cap,teacher):
    from stvg.initialization import evidence_hash
    e=seal['files'][row['id']];f=root/e['path'];r=root/e['receipt_path'];assert sha(f)==e['sha256']and sha(r)==e['receipt_sha256']
    with np.load(f)as z:
        assert np.array_equal(cap['Source'].numpy(),z['Source'])and np.array_equal(cap['frame_ids'],z['frame_ids'])
        assert np.array_equal(cap['Native_interval'],z['Native_interval'])
        assert np.array_equal(cap['positions8'],z['positions8'])
    receipt=json.loads(r.read_text());assert evidence_hash(teacher['boxes'].numpy())==receipt['initialization']['evidence_sha256']


def capture_one(model,row,teacher_service,plan,oldroot,oldseal):
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    from stvg.positions import observation_positions
    from stvg.initialization import initialize
    from tools.boxfixed_tube.adapter import capture
    from tools.boxfixed_tube.evidence import SourceEvidence
    tick=time.perf_counter();frames,fids=decode(row['metadata'],row['dataset']);validate_input(frames,fids,row['metadata']);decode_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    cap=capture(model,frames,fids,row['metadata']['caption']);torch.cuda.synchronize();source_seconds=time.perf_counter()-tick;positions=observation_positions(fids,cap['native_extent']);cap['positions8']=positions
    teacher,cost=teacher_service(frames[positions],row['metadata']['caption']);verify_old(oldroot,oldseal,row,cap,teacher)
    raw,initialization=initialize(cap['Source'].double().numpy(),positions,teacher['boxes'].double().numpy(),plan['datasets'][row['dataset']]['configuration']['E_calibration'])
    tick=time.perf_counter();evidence=SourceEvidence.from_native_masks(cap['features'],cap['valid'],fids,positions,teacher['mask_logits'],teacher['teacher_image_size'],cap['grid_shape'],selected_query=teacher['selected_query'])
    result=dict(Source=cap['Source'].numpy(),raw=raw,Native_interval=cap['Native_interval'],Global6_interval=cap['Global6_interval'],frame_ids=fids,positions8=positions,evidence=evidence,native_boxes=teacher['boxes'].double(),request_sha256=digest(row),initialization=initialization)
    receipt=dict(Source_Native_observations_teacher_bitexact=True,feature_interface=cap['feature_interface'],temporal_interface=cap['temporal_interface'],source_checkpoint_sha256=model.load_audit['checkpoint_sha256'],Source224_decoder_calls=1,Source448_calls=0,optimizer_updates=0,teacher_queries=0 if cost['cache_hit']else 1,teacher_cache_key=cost['cache_key'],teacher_cache_hit=cost['cache_hit'],cost=dict(decode_seconds=decode_seconds,Source_seconds=source_seconds,teacher_seconds=cost['seconds'],seed_seconds=time.perf_counter()-tick),GT_read=False,request_sha256=digest(row),Global6_interval=cap['Global6_interval'])
    return result,receipt


def geometry(c,p,ds):
    from tools.boxfixed_tube.spatial import build_geometry
    from stvg.predictor import output_writeback
    E=torch.from_numpy(output_writeback(c['Source'],c['raw'],baseline(ds)['gamma']));return build_geometry(c['evidence'],E,c['native_boxes'])


def gpu_capture(ds,full=False):
    setup();p,rows=records(ds,full);lock=(ROOT/'runtime/gpu.lock').open('a');print('Waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
    from stvg.tubedetr import load_source
    from stvg.tensors import state_hash
    from tools.boxfixed_tube.teacher_service import Teacher
    model=load_source(p['datasets'][ds]['configuration']);state_before=state_hash(model.state_dict());oldroot,oldseal=old_reference(ds,p);teacher=Teacher(OUT/'teacher_cache',p['teacher_fingerprint']);entries={};target=OUT/ds/('full'if full else'captures');target.mkdir(parents=True,exist_ok=True)
    if full:
        selected=json.loads((PUBLIC/ds/'selected_profile.json').read_text());assert selected['plan_sha256']==sha(PLAN);params=selected['parameters'];devroot=OUT/ds/'profiles'/selected['profile_id'];devseal=json.loads((devroot/'sealed.json').read_text());assert sha(devroot/'sealed.json')==selected['dev_prediction_seal_sha256']
    try:
        for i,row in enumerate(rows):
            q=row['id'];path=target/(q+('.npz'if full else'.pt'));rp=path.with_suffix('.json')
            if path.exists()and rp.exists():
                meta=json.loads(rp.read_text());assert meta['plan_sha256']==sha(PLAN)and meta['request_sha256']==digest(row)and meta['payload_sha256']==sha(path)
                if full:assert meta['selected_profile_id']==selected['profile_id']
                entries[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp));continue
            if full and q in devseal['files']:
                e=devseal['files'][q];assert sha(devroot/e['path'])==e['sha256'];proof=json.loads((devroot/e['receipt_path']).read_text());assert proof['request_sha256']==digest(row);shutil.copy2(devroot/e['path'],path);meta=dict(proof,selected_profile_id=selected['profile_id'],reused_development=True)
            else:
                tick=time.perf_counter();c,meta=capture_one(model,row,teacher,p,oldroot,oldseal)
                if full:
                    spatial_tick=time.perf_counter();g=geometry(c,p,ds);r=predict(g,c['evidence'],c['Source'],c['raw'],c['Global6_interval'],params);rn=predict(g,c['evidence'],c['Source'],c['raw'],c['Native_interval'],params)
                    save_npz(path,c,r,rn);meta.update(selected_profile_id=selected['profile_id'],spatial_seconds=time.perf_counter()-spatial_tick,selected_parameters=params,reused_development=False)
                else:torch.save(c,path)
                meta['total_query_seconds']=time.perf_counter()-tick
            meta.update(plan_sha256=sha(PLAN),request_sha256=digest(row),payload_sha256=sha(path));write(rp,meta);entries[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp))
            write(OUT/'worker_progress.json',dict(dataset=ds,stage='full'if full else'capture128',complete=i+1,total=len(rows),labels_read=False));print(ds,'full'if full else'capture128',i+1,'/',len(rows),flush=True)
        assert state_hash(model.state_dict())==state_before and all(not x.requires_grad and x.grad is None for x in model.parameters())
        write(target/'sealed.json',dict(status='complete',dataset=ds,queries=len(rows),files=entries,plan_sha256=sha(PLAN),manifest_sha256=p['datasets'][ds]['full_manifest_sha256'if full else'manifest_sha256'],labels_read=False,model_weights_unchanged=True,profile=selected if full else None))
    finally:teacher.close();lock.close()


def save_npz(path,c,r,rn):
    from stvg.predictor import output_writeback
    # E comes from the selected geometry, encoded by raw and parameters in receipt.
    np.savez(path,Source=c['Source'],BoxFixed=r['final'].numpy(),BoxFixedNative=rn['final'].numpy(),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']),Native_interval=np.asarray(c['Native_interval']),Global6_interval=np.asarray(c['Global6_interval']),actions=r['actions'].numpy(),confidence=r['reliability'].numpy())


def capture_seal(ds):
    target=OUT/ds/'captures';s=json.loads((target/'sealed.json').read_text());assert s['status']=='complete'and s['queries']==128 and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(target/e['path'])==e['sha256']and sha(target/e['receipt_path'])==e['receipt_sha256']
    return target,s


def profile_worker(ds,request):
    setup();p,rows=records(ds);torch.set_num_threads(1);params=json.loads(Path(request).read_text());target=OUT/ds/'profiles'/profile_id(params);target.mkdir(parents=True,exist_ok=True);root,s=capture_seal(ds);entries={}
    from tools.boxfixed_tube.spatial import evidence_dp
    for i,row in enumerate(rows):
        q=row['id'];path=target/(q+'.npz');rp=path.with_suffix('.json');c=torch.load(root/s['files'][q]['path'],map_location='cpu',weights_only=False);assert c['request_sha256']==digest(row);g=geometry(c,p,ds);r=predict(g,c['evidence'],c['Source'],c['raw'],c['Global6_interval'],params);rn=predict(g,c['evidence'],c['Source'],c['raw'],c['Native_interval'],params)
        if params==baseline(ds):
            direct=evidence_dp(g.for_interval(c['Global6_interval']));assert torch.equal(direct['final'],r['final'])and direct['score']==r['score'],'Default port differs from upstream'
        save_npz(path,c,r,rn);write(rp,dict(plan_sha256=sha(PLAN),request_sha256=digest(row),profile_id=profile_id(params),parameters=params,admitted_frames=r['candidate_admitted_frames'],changed_from_ERef_frames=r['changed_from_ERef_frames'],new_GPU_forwards=0,labels_read=False));entries[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),parameters=params,capture_seal_sha256=sha(root/'sealed.json'),labels_read=False));print(ds,'profile',params,flush=True)


def score(ds,params=None,full=False):
    p,rows=records(ds,full);target=OUT/ds/('full'if full else'profiles/'+profile_id(params));s=json.loads((target/'sealed.json').read_text());assert s['status']=='complete'and s['queries']==len(rows)and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():assert sha(target/e['path'])==e['sha256']and sha(target/e['receipt_path'])==e['receipt_sha256']
    if full:params=s['profile']['parameters']
    label_key='full_labels'if full else'labels';assert sha(ROOT/p['datasets'][ds][label_key])==p['datasets'][ds][label_key+'_sha256']
    truth={r['id']:r for r in map(json.loads,(ROOT/p['datasets'][ds]['full_labels'if full else'labels']).read_text().splitlines())}
    from stvg.metrics import DenseTube
    details=[];costs=[];admitted=[];changes=[]
    for row in rows:
        q=row['id'];gt=truth[q];receipt=json.loads((target/s['files'][q]['receipt_path']).read_text());costs.append(receipt);admitted.append(receipt.get('admitted_frames'));changes.append(receipt.get('changed_from_ERef_frames'));boxes={int(k):v for k,v in gt['boxes'].items()};e=s['files'][q]
        with np.load(target/e['path'])as z:
            meta=dict(frame_ids=z['frame_ids'].tolist(),input=row['metadata']);values={};unseen={};observed={};obs=set(z['frame_ids'][z['positions8']].tolist())
            for name,b,interval in [('Source',z['Source'],z['Native_interval']),('Spatial',z['BoxFixedNative'],z['Native_interval']),('Temporal',z['Source'],z['Global6_interval']),('Full',z['BoxFixed'],z['Global6_interval'])]:
                m=DenseTube(b,meta,boxes,gt['interval'],clip=ds=='hc2');values[name]=m.score(interval);
                if len(details)<3:
                    from stvg.metrics import official
                    checked_metric=official(b,meta,boxes,gt['interval'],interval,ds)
                    assert all(abs(values[name][key]-checked_metric[key])<1e-10 for key in ['v','s','t']),(ds,name,values[name],checked_metric)
                mask=np.array([int(t)not in obs for t in m.fids]);unseen[name]=float(m.iou[mask].mean())if mask.any()else None;observed[name]=float(m.iou[~mask].mean())if(~mask).any()else None
        details.append(dict(id=q,parent=gt['parent'],values=values,unobserved=unseen,observed=observed))
    def mean(v):v=[x for x in v if x is not None];return float(np.mean(v))if v else None
    result=dict(status='complete',dataset=ds,queries=len(rows),parameters=params,means={name:{k:100*mean([d['values'][name][k]for d in details])for k in ['v','s','t']}for name in ['Source','Spatial','Temporal','Full']},unobserved_sIoU={name:100*mean([d['unobserved'][name]for d in details])for name in ['Source','Spatial','Temporal','Full']},observed_sIoU={name:100*mean([d['observed'][name]for d in details])for name in ['Source','Spatial','Temporal','Full']},Full_minus_Temporal_pp=100*mean([d['values']['Full']['v']-d['values']['Temporal']['v']for d in details]),loss_over5pp_queries=sum(d['values']['Full']['v']<d['values']['Temporal']['v']-.05 for d in details),prediction_seal_sha256=sha(target/'sealed.json'),all_predictions_sealed_before_labels=True,mean_admitted_frames=mean(admitted),mean_changed_from_ERef_frames=mean(changes),not_independent_confirmation=True)
    write(target/'details_private.json',details);write(target/'metrics.json',result)
    if full:
        from tools.tubedetr_opd_direct_diagnosis import cluster_difference
        result['paired_difference']=cluster_difference([d['values']['Full']['v']-d['values']['Source']['v']for d in details],[d['parent']for d in details]);parents=sorted({d['parent']for d in details});result['equal_parent_Full_m_vIoU']=100*mean([mean([d['values']['Full']['v']for d in details if d['parent']==a])for a in parents]);result['parents']=len(parents);write(PUBLIC/ds/'full_metrics.json',result)
        (PUBLIC/ds/'report.md').write_text('# TubeDETR BoxFixed＋Global6：选定后完整总体\n\n128开发曝光和全量重叠，全量不再用于参数选择。无OPD/训练/448。Spatial为BoxFixed＋Native，Temporal为Source＋Global6，Full为BoxFixed＋Global6。旧五行消融保持，不将无OPD算法命名WithoutOPD。缓存部署成本不能冒充新端到端成本。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        write(PUBLIC/ds/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/ds/'selected_profile.json',PUBLIC/ds/'decisions.json',PUBLIC/ds/'full_metrics.json',PUBLIC/ds/'report.md']},prediction_seal_sha256=sha(target/'sealed.json'),publication_pending=True))
    return result


def child(stage,ds,request=None):
    command=[sys.executable,'-B','-u','tools/tubedetr_boxfixed_development.py','--stage',stage,'--dataset',ds]
    if request:command+=['--request',str(request)]
    log=ROOT/'logs'/('tubedetr_boxfixed_'+ds+'_'+stage+'.log');env=dict(os.environ)
    if stage not in ['capture','full']:env['CUDA_VISIBLE_DEVICES']=''
    write(OUT/'status.json',dict(status=stage,dataset=ds,request=str(request)if request else None))
    with log.open('a')as f:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env);path=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(path.read_text());j['jobs']['boxfixed_worker']=dict(pid=proc.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage,dataset=ds);write(path,j)
        if proc.wait():raise RuntimeError('BoxFixed worker failed '+ds+' '+stage)


def controller():
    p=checked();all_profiles={}
    for ds in p['dataset_order']:
        child('capture',ds);decisions=[];tested={};base=baseline(ds);current=base.copy();pub=PUBLIC/ds;pub.mkdir(parents=True,exist_ok=True)
        def evaluate(params):
            key=profile_id(params)
            if key not in tested:
                req=OUT/ds/'requests'/(key+'.json');write(req,params);child('profile',ds,req);child('score',ds,req);tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text())
            return tested[key]
        def choose():
            return max(tested.values(),key=lambda m:(m['means']['Full']['v'],m['parameters']==base,-abs(m['parameters']['gamma']-base['gamma'])))['parameters'].copy()
        evaluate(base)
        specs=[('initialization','gamma',p['gamma_grid'],'先隔离共识初始化强度；新方法gamma0仍包含BoxFixed传播，不是Temporal-only。'),('admission','admission',p['admission_grid'],'比较往返可靠性准入，检查传播覆盖与负例，不把置信当正确概率。'),('protection','guard',p['guard_grid'],'检查替代ERef的幅度风险；固定证据变换不改变。'),('association','link',p['link_grid'],'检查连接项提供的定位信息与错误延续；零权重作为公开控制。')]
        for name,key,grid,reason in specs:
            start=current.copy();batch=[evaluate({**start,key:value})for value in grid];current=choose();win=evaluate(current);reference_metrics=evaluate(base);interpretation=dict(selected_value=current[key],original_value=base[key],Full_m_vIoU_gain_vs_original_pp=win['means']['Full']['v']-reference_metrics['means']['Full']['v'],dense_sIoU_gain_vs_original_pp=win['means']['Full']['s']-reference_metrics['means']['Full']['s'],unobserved_sIoU_gain_vs_original_pp=win['unobserved_sIoU']['Full']-reference_metrics['unobserved_sIoU']['Full'],tail_loss_queries=win['loss_over5pp_queries'],admitted_frames=win['mean_admitted_frames'],changed_from_ERef_frames=win['mean_changed_from_ERef_frames'],conclusion=('原参数保留；本阶段未得到更好的完整128定位证据'if current==base else'数值干预在固定128上更优；同时公开空间/未观察和尾损，不外推全量'))
            entry=dict(stage=name,reason_before=reason,start=start,all_candidates=batch,selected=current,selected_metrics=win,observed_interpretation=interpretation,next_reason='保留全128 m_vIoU最优全局参数，继续冻结的下一独立因素；空间、观察/未观察、尾损均公开，不用伪目标分数代替GT定位');decisions.append(entry);write(pub/'decisions.json',decisions)
        gamma=p['gamma_grid'];i=gamma.index(current['gamma']);neighbors=[gamma[j]for j in [i-1,i+1]if 0<=j<len(gamma)];batch=[evaluate({**current,'gamma':x})for x in neighbors];current=choose();decisions.append(dict(stage='coupling_recheck',reason_before='仅复查所选新优化参数下相邻gamma，最多两个，不扩大预算',all_candidates=batch,selected=current));assert len(tested)<=p['max_profiles_per_dataset'];write(pub/'decisions.json',decisions)
        winner=evaluate(current);selection=dict(status='locked_before_full',dataset=ds,parameters=current,profile_id=profile_id(current),plan_sha256=sha(PLAN),dev_metrics=winner,profiles_tested=len(tested),dev_prediction_seal_sha256=sha(OUT/ds/'profiles'/profile_id(current)/'sealed.json'),selection_objective='fixed128 equal-query Full m_vIoU; original baseline always eligible, exact ties retain baseline',full_not_used_for_selection=True,development_exposed=True);write(pub/'selected_profile.json',selection);all_profiles[ds]=selection
    # Both128 selections are immutable before either full population starts.
    write(PUBLIC/'selected_profiles.json',all_profiles);write(OUT/'status.json',dict(status='selected_both128_starting_full',profiles={ds:v['parameters']for ds,v in all_profiles.items()}))
    for ds in p['dataset_order']:child('full',ds);child('full_score',ds)
    files=[PLAN,PUBLIC/'selected_profiles.json',*[PUBLIC/ds/'final_seal.json'for ds in p['dataset_order']]];write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True));write(OUT/'status.json',dict(status='complete',both_directions=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['capture','profile','score','full','full_score']);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
    try:
        if args.stage=='capture':gpu_capture(args.dataset)
        elif args.stage=='full':gpu_capture(args.dataset,True)
        elif args.stage=='profile':profile_worker(args.dataset,args.request)
        elif args.stage=='score':score(args.dataset,json.loads(Path(args.request).read_text()))
        elif args.stage=='full_score':score(args.dataset,full=True)
        else:controller()
    except Exception as e:
        v=dict(status='failed',stage=args.stage,dataset=args.dataset,error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
