"""Source-only fitting; sealed no-GT target gates; separate scores and audits."""
from common import *
from reliability_source_worker import C,check_source,source_cases,raw_inputs,cohort_row
from reliability_signals import signals,risks
import reliability_math as math
import argparse,ast,csv
RULES=['expert_risk','Source_confidence','combined'];COV=[.1,.25,.5]
POLICIES=['Source','Full','RiskGate','RandomGate']+[f'{r}_{int(c*100)}'for c in COV for r in RULES]
def checked_source(target,key,raw=False):
 case=next(x for x in source_cases(target)if x['id']==key);dest=C/'source'/target;e=read(dest/'receipts'/(key+'.json'))
 for f,h in [('path','sha256'),('compact_path','compact_sha256'),('teacher_path','teacher_sha256'),('request_path','request_sha256')]:assert sha(D/e[f])==e[h]
 assert e['input_binding']['source_case']==case and e['new_Source']==e['new_optimizer']==0 and e['new_ReferDINO_queries']==1
 native=torch.load(case['native_Source_path'],map_location='cpu',weights_only=False,mmap=True);assert sha(case['native_Source_path'])==case['native_Source_sha256'];teacher=torch.load(D/e['teacher_path'],map_location='cpu',weights_only=False,mmap=True)
 assert teacher['native_same_track_all8']and teacher['nested_mask_all_false']and int(np.argmax(np.asarray(teacher['native_query_scores'])))==int(teacher['selected_query'])
 request=torch.load(D/e['request_path'],map_location='cpu',weights_only=False,mmap=True);assert hashlib.sha256(request['frames'].tobytes()).hexdigest()==case['context']['RGB_sha256']
 if raw:
  row,again,frames=raw_inputs(target,case);assert np.array_equal(frames[case['positions8']],request['frames'])and request['caption']==row['metadata']['caption']
 a,v=math.build(native,teacher,target,case['positions8']);b,w=math.build(native,teacher,target,case['positions8']);cold=math.replay(torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True));assert set(a)==set(b)==set(cold)and all(np.array_equal(x,b[k])and np.array_equal(x,cold[k])for k,x in a.items())
 with np.load(D/e['path'])as z:assert set(z.files)==set(a)and all(np.array_equal(x,z[k])for k,x in a.items())
 feature=signals(native['Source'],native['native_logits'],native['records'],native['frame_ids'],case['positions8'],teacher,fixed(R/'configs/default.json')['datasets'][target]['E_calibration']);assert feature==e['signals']
 return dict(prediction_sha256=e['sha256'],native_Source_sha256=case['native_Source_sha256'],teacher_sha256=e['teacher_sha256'],correct_Source_policy_RGB_and_8expert_and_repool_geometry_posterior_finalDP_cold_exact=True,actual_fullRGB_checked=raw)
def source_bridge(target):
 p=check_source();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);dest=C/'source'/target;w=read(dest/'bridge_input.json');assert len(w['keys'])==2 and not(dest/'actual_bridge.json').exists();cases=[checked_source(target,k,True)for k in w['keys']]
 atomic(dest/'actual_bridge.json',dict(status='passed_two_actual_new_source_expert_correct_checkpoint_8positions_RGB_payload_fullmath_allarrays_cold_exact',cases=cases,plan_fingerprint=p['fingerprint'],worker=identity(),new_Source=0,new_expert=0,new_GT=0));print(target,'actual source expert first2 bridge passed',flush=True)
def source_input_audit():
 check_source();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);seal=read(C/'source_sealed.json');assert seal['counts']['committed']==72;start=time.perf_counter();cases={}
 for target in ['hc2','vidstg']:
  ss=read(C/'source'/target/'sealed.json');assert ss['queries']==36 and seal['domains'][target]['seal_sha256']==sha(C/'source'/target/'sealed.json');cases[target]=[checked_source(target,k)for k in ss['files']]
 atomic(C/'source_input_audit.json',dict(status='all72_new_source_expert_fullmath_signals_inputSHA_exact',cases=cases,seal_sha256=sha(C/'source_sealed.json'),worker=identity(),new_GT=0,elapsed_wall_seconds=time.perf_counter()-start));print('all72 source input/math audit passed',flush=True)
def manual_metric(boxes,ids,iv,meta,g,ds):
 b=np.asarray(boxes,np.float64);pix=np.column_stack([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2])
 if ds=='hc2':pix=np.maximum(pix,0)
 gt={int(k):v for k,v in g['boxes'].items()};ff=np.array(sorted(gt));p=np.column_stack([np.interp(ff,ids,pix[:,j])for j in range(4)]);truth=np.asarray([gt[int(f)]for f in ff])/np.array([meta['width'],meta['height']]*2);inter=np.maximum(np.minimum(p[:,2:],truth[:,2:])-np.maximum(p[:,:2],truth[:,:2]),0).prod(1);den=np.maximum(p[:,2:]-p[:,:2],0).prod(1)+np.maximum(truth[:,2:]-truth[:,:2],0).prod(1)-inter;iou=np.divide(inter,den,out=np.zeros(len(ff)),where=den>0);iou[(ff<ids[0])|(ff>ids[-1])]=0;a,b=map(int,iv);c,d=map(int,g['interval']);over=max(0,min(b,d)-max(a,c));v=float(iou[(ff>=max(a,c))&(ff<min(b,d))].sum()/max(max(b,d)-min(a,c),1));return dict(v=v,t=over/(b-a+d-c-over),R03=float(v>.3),R05=float(v>.5))
def official_metric(boxes,ids,iv,meta,g,ds):
 from stvg.metrics import official
 v=official(boxes,dict(input=meta,frame_ids=list(map(int,ids))),{int(k):v for k,v in g['boxes'].items()},g['interval'],iv,ds);return dict(v=v['v'],t=v['t'],R03=float(v['v']>.3),R05=float(v['v']>.5))
def source_derive(manual=False):
 assert read(C/'source_input_audit.json')['seal_sha256']==sha(C/'source_sealed.json');from studies.map_paper_completion_v1.common import labels
 out={};metric=manual_metric if manual else official_metric
 for target in ['hc2','vidstg']:
  cases=source_cases(target);ds=cases[0]['source_domain'];truth=labels(ds);rr=[]
  for case in cases:
   key=case['id'];row=cohort_row(target,key);e=read(C/'source'/target/'receipts'/(key+'.json'));g=truth[key];assert sha(D/e['path'])==e['sha256']
   with np.load(D/e['path'])as z:values={arm:metric(z[arm],z['frame_ids'],z[arm+'_interval'],row['metadata'],g,ds)for arm in ['Source','Full']}
   rr.append(dict(id=key,dataset=ds,parent=str(g['parent']),signals=e['signals'],values=values))
  out[target]=rr
 return out
def source_score():
 check_source();torch.set_num_threads(1);rr=source_derive();atomic(C/'source_metrics.json',dict(domains=rr));atomic(C/'source_score_receipt.json',dict(status='source_only144_official_checks_complete',queries=72,official_checks=144,metrics_sha256=sha(C/'source_metrics.json'),worker=identity()));print('source-only official scores complete',flush=True)
def source_score_audit():
 check_source();torch.set_num_threads(1);rr=source_derive(True);old=read(C/'source_metrics.json')['domains'];mx=0.
 for target in rr:
  for a,b in zip(rr[target],old[target]):
   assert all(a[k]==b[k]for k in ['id','dataset','parent','signals'])
   for arm in ['Source','Full']:
    for f,x in a['values'][arm].items():mx=max(mx,abs(x-b['values'][arm][f]));assert abs(x-b['values'][arm][f])<2e-10
 atomic(C/'source_score_audit.json',dict(status='source_only_independent_normalized_pixel_scores_parent_and_input_means_exact',queries=72,manual_checks=144,max_discrepancy=mx,metrics_sha256=sha(C/'source_metrics.json'),worker=identity()));print('source-only independent score audit passed',flush=True)
def keyhash(dataset,key,tag='risk_tie'):return hashlib.sha256(f'{tag}|20261105|{dataset}|{key}'.encode()).hexdigest()
def cutoff(train,rule,coverage):
 ff=[r['signals']for r in train];order=sorted([(risks(ff,r['signals'])[rule],keyhash(r['dataset'],r['id']))for r in train],key=lambda a:(-a[0],a[1]));return order[max(0,int(np.ceil(len(order)*coverage))-1)]
def reject(score,h,cut):return score>cut[0]or(score==cut[0]and h<=cut[1])
def candidates():return [dict(name='alwaysFull',rule=None,coverage=0.)]+[dict(name=f'{r}_{int(c*100)}',rule=r,coverage=c)for c in COV for r in RULES]
def fit():
 check_source();assert read(C/'source_score_audit.json')['metrics_sha256']==sha(C/'source_metrics.json');rr=read(C/'source_metrics.json')['domains'];result={}
 for target,source in rr.items():
  parents=sorted({r['parent']for r in source},key=lambda p:keyhash(target,p,'source_fold'));fold={p:i%4 for i,p in enumerate(parents)};configs=[]
  for cfg in candidates():
   scored=[]
   for k in range(4):
    train=[r for r in source if fold[r['parent']]!=k];valid=[r for r in source if fold[r['parent']]==k];ff=[r['signals']for r in train];cut=cutoff(train,cfg['rule'],cfg['coverage'])if cfg['rule']else None
    for r in valid:
     flag=reject(risks(ff,r['signals'])[cfg['rule']],keyhash(r['dataset'],r['id']),cut)if cut else False;scored.append(dict(id=r['id'],parent=r['parent'],fold=k,reject_to_Source=flag,values=r['values']['Source'if flag else'Full']))
   assert len(scored)==36;configs.append(dict(**cfg,mean_v=float(np.mean([v['values']['v']for v in scored])),mean_t=float(np.mean([v['values']['t']for v in scored])),rejected=sum(v['reject_to_Source']for v in scored),OOF_cases=scored))
  selected=max(range(len(configs)),key=lambda i:configs[i]['mean_v']);cfg=configs[selected];cut=cutoff(source,cfg['rule'],cfg['coverage'])if cfg['rule']else None
  result[target]=dict(selected_name=cfg['name'],selected_rule=cfg['rule'],selected_coverage=cfg['coverage'],cutoff=cut,source_signals=[r['signals']for r in source],source_parent_count=len(parents),fold_by_parent=fold,all_configs=configs,all_source_mean_Source_v=float(np.mean([r['values']['Source']['v']for r in source])),all_source_mean_Full_v=float(np.mean([r['values']['Full']['v']for r in source])),no_target_labels_for_selection=True)
 atomic(C/'source_selection.json',dict(status='source_only_four_parentfold_gate_selection_frozen_before_target_decisions',domains=result,predeclaration_sha256=sha(C/'rule_predeclaration.json'),source_audit_sha256=sha(C/'source_score_audit.json'),worker=identity()));print({t:dict(selected_name=x['selected_name'],OOFv=x['all_configs'][next(i for i,c in enumerate(x['all_configs'])if c['name']==x['selected_name'])]['mean_v'],parents=x['source_parent_count'])for t,x in result.items()},flush=True)
def target_feature(ds,row):
 from stvg.readout import prediction
 from stvg.positions import observation_positions
 a,t,e=reference(ds,row);b=e['input_binding'];assert sha(b['parent']['capture'])==b['parent']['capture_sha256'];native=torch.load(b['parent']['capture'],map_location='cpu',weights_only=False,mmap=True);pos=observation_positions(native['frame_ids'],prediction(native['native_logits'],torch.as_tensor(native['Source']),native['records'],native['frame_ids'])['indices']);assert pos==a['positions8'].tolist()
 return a,signals(native['Source'],native['native_logits'],native['records'],native['frame_ids'],pos,t,fixed(R/'configs/default.json')['datasets'][ds]['E_calibration']),dict(reference_prediction_sha256=e['sha256'],teacher_sha256=e['current_teacher']['payload_sha256'],native_Source_capture_sha256=b['parent']['capture_sha256'])
def target_predict():
 p=check_source();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);selection=read(C/'source_selection.json');domains={};start=time.perf_counter()
 for ds in ['hc2','vidstg']:
  cfg=selection['domains'][ds];source=fixed(C/'source_metrics.json')['domains'][ds];ff=cfg['source_signals'];features={};feature_times={};cuts={c['name']:cutoff(source,c['rule'],c['coverage'])for c in candidates()if c['rule']};flags={name:{}for name in POLICIES};files={}
  for row in rows(ds):
   resource(start);key=row['id'];tick=time.perf_counter();a,f,binding=target_feature(ds,row);rs=risks(ff,f);feature_times[key]=time.perf_counter()-tick;features[key]=dict(signals=f,risks=rs,input_binding=binding)
   flags['Source'][key]=True;flags['Full'][key]=False;flags['RiskGate'][key]=reject(rs[cfg['selected_rule']],keyhash(ds,key),cfg['cutoff'])if cfg['selected_rule']else False
   for c in candidates():
    if c['rule']:flags[c['name']][key]=reject(rs[c['rule']],keyhash(ds,key),cuts[c['name']])
  k=sum(flags['RiskGate'].values());random=set(sorted(features,key=lambda key:keyhash(ds,key,'random_control'))[:k]);flags['RandomGate']={key:key in random for key in features};assert sum(flags['RandomGate'].values())==k
  for row in rows(ds):
   key=row['id'];a,_,_=reference(ds,row);out=dict(frame_ids=a['frame_ids'],positions8=a['positions8'])
   for name in POLICIES:
    src=flags[name][key];out[name]=a['Source'if src else'Ours'];out[name+'_interval']=a['Source_interval'if src else'Ours_interval']
   dest=C/'target'/ds/'predictions'/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb')as f:np.savez_compressed(f,**out)
   with np.load(dest)as z:assert all(np.array_equal(v,z[name])for name,v in out.items())
   files[key]=dict(path=str(dest.relative_to(D)),sha256=sha(dest),features=features[key],cached_signal_wall_seconds=feature_times[key],reject_flags={name:flags[name][key]for name in POLICIES},GT_read=False)
  atomic(C/'target'/ds/'sealed.json',dict(queries=128,files=files,source_selection_sha256=sha(C/'source_selection.json'),random_exact_rejected_coverage=k,policies=POLICIES,GT_read=False,plan_fingerprint=p['fingerprint'],worker=identity()));domains[ds]=dict(queries=128,seal_sha256=sha(C/'target'/ds/'sealed.json'))
 atomic(C/'target_sealed.json',dict(status='all256_target_decisions_and_fullarrays_sealed_before_targetlabels',domains=domains,source_selection_sha256=sha(C/'source_selection.json'),GT_read=False,worker=identity()));print('all256 target gates sealed, targetGT not read',flush=True)
def target_input_audit():
 check_source();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);ss=read(C/'target_sealed.json');selection=read(C/'source_selection.json');checked=0
 for ds in ['hc2','vidstg']:
  seal=read(C/'target'/ds/'sealed.json');assert ss['domains'][ds]['seal_sha256']==sha(C/'target'/ds/'sealed.json');cfg=selection['domains'][ds];source=fixed(C/'source_metrics.json')['domains'][ds];cuts={c['name']:cutoff(source,c['rule'],c['coverage'])for c in candidates()if c['rule']};primary={}
  for row in rows(ds):
   key=row['id'];e=seal['files'][key];a,f,b=target_feature(ds,row);rs=risks(cfg['source_signals'],f);assert e['features']==dict(signals=f,risks=rs,input_binding=b);primary[key]=reject(rs[cfg['selected_rule']],keyhash(ds,key),cfg['cutoff'])if cfg['selected_rule']else False
   assert e['reject_flags']['RiskGate']==primary[key]and e['reject_flags']['Source']and not e['reject_flags']['Full'];assert sha(D/e['path'])==e['sha256']
   for c in candidates():
    if c['rule']:assert e['reject_flags'][c['name']]==reject(rs[c['rule']],keyhash(ds,key),cuts[c['name']])
   with np.load(D/e['path'])as z:
    for name in POLICIES:
     src=e['reject_flags'][name];assert np.array_equal(z[name],a['Source'if src else'Ours'])and np.array_equal(z[name+'_interval'],a['Source_interval'if src else'Ours_interval']);checked+=1
  k=sum(primary.values());random=set(sorted(primary,key=lambda key:keyhash(ds,key,'random_control'))[:k]);assert all(e['reject_flags']['RandomGate']==(key in random)for key,e in seal['files'].items())
 atomic(C/'target_input_audit.json',dict(status='all256_source_only_gate_allarray_input_features_decisions_and_random_exact_coverage_independently_rederived',array_checks=checked,target_seal_sha256=sha(C/'target_sealed.json'),worker=identity(),new_GT=0));print('all256 target gate input/coldarray audit passed',flush=True)
def target_derive(manual=False):
 assert read(C/'target_input_audit.json')['target_seal_sha256']==sha(C/'target_sealed.json');from studies.map_paper_completion_v1.common import labels
 out={};metric=manual_metric if manual else official_metric
 for ds in ['hc2','vidstg']:
  truth=labels(ds);seal=read(C/'target'/ds/'sealed.json');rr=[]
  for row in rows(ds):
   key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key]
   with np.load(D/e['path'])as z:
    # Score each unique branch once; every policy is an exact sealed selection
    # of that whole Source/Full tube, independently checked above.
    branches={a:metric(z[a],z['frame_ids'],z[a+'_interval'],row['metadata'],g,ds)for a in ['Source','Full']};values={a:branches['Source'if e['reject_flags'][a]else'Full']for a in POLICIES}
   rr.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values,reject_flags=e['reject_flags'],risks=e['features']['risks']))
  out[ds]=rr
 return out
def target_score():
 check_source();torch.set_num_threads(1);rr=target_derive();atomic(C/'target_metrics.json',dict(domains=rr));atomic(C/'target_score_receipt.json',dict(status='all256_source_Full_unique_branches_officially_scored_and_sealed_gate_scores_exact_selected',unique_official_checks=512,policies=len(POLICIES),metrics_sha256=sha(C/'target_metrics.json'),worker=identity()));print('target gate official scores complete',flush=True)
def target_score_audit():
 check_source();torch.set_num_threads(1);rr=target_derive(True);old=read(C/'target_metrics.json')['domains'];mx=0.
 for ds in rr:
  for a,b in zip(rr[ds],old[ds]):
   assert all(a[k]==b[k]for k in ['id','parent','duration_group','reject_flags','risks'])
   for arm in POLICIES:
    for f,v in a['values'][arm].items():mx=max(mx,abs(v-b['values'][arm][f]));assert abs(v-b['values'][arm][f])<2e-10
 atomic(C/'target_score_audit.json',dict(status='all256_sealed_gate_manual_normalized_pixel_scores_input_parent_allmeans_audited',unique_manual_checks=512,max_discrepancy=mx,metrics_sha256=sha(C/'target_metrics.json'),worker=identity()));print('target gate independent manual score audit passed',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('task',choices=['source_bridge','source_input_audit','source_score','source_score_audit','fit','target_predict','target_input_audit','target_score','target_score_audit']);a.add_argument('--dataset',choices=['hc2','vidstg']);x=a.parse_args();globals()[x.task](*([x.dataset]if x.task=='source_bridge'else[]))
