"""Read saved HC64 traces and already scored anonymous metrics; no model or GT annotations."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import collections,hashlib,json,math,pathlib,time
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=pathlib.Path(__file__).resolve().parents[2];TR=ROOT/'outputs/state_opd_hc128';OUT=TR/'monitor'
ARMS=['StateOPD','Direct','DirectMC','DirectFullRef'];TAU=.01

def rd(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def arr(v):return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
def stats(x):
 a=np.asarray(x,dtype=float).reshape(-1);a=a[np.isfinite(a)]
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),minimum=float(a.min()),maximum=float(a.max())) if len(a) else dict(n=0)
def corners(b):return np.concatenate([b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2],-1)
def chartd(b,e):
 dz=np.concatenate([(b[...,:2]-e[...,:2])/e[...,2:],np.log(b[...,2:])-np.log(e[...,2:])],-1)
 return np.linalg.norm(dz,axis=-1)/2

def motion(b,e,obs):
 co,ce=corners(b),corners(e);inter=np.maximum(0,np.minimum(co[:,2:],ce[:,2:])-np.maximum(co[:,:2],ce[:,:2])).prod(-1)
 iou=inter/(b[:,2:].prod(-1)+e[:,2:].prod(-1)-inter)
 center=np.linalg.norm(b[:,:2]-e[:,:2],axis=-1);d=chartd(b,e)
 return dict(corner_abs_coordinate_change_pct_image=float(100*np.abs(co-ce).mean()),
             center_L2_change_in_normalized_image_units=float(center.mean()),
             center_L2_change_observed=float(center[obs].mean()),center_L2_change_unobserved=float(center[~obs].mean()),
             standardized_chart_d_mean=float(d.mean()),standardized_chart_d_max=float(d.max()),
             same_node_IoU_vs_E_mean=float(iou.mean()),same_node_IoU_vs_E_min=float(iou.min()),
             nodes_IoU_vs_E_ge099=float((iou>=.99).mean()),nodes_IoU_vs_E_ge095=float((iou>=.95).mean()))

def require(c,msg):
 if not c:raise AssertionError(msg)

start=time.perf_counter();lock=rd(TR/'lock.json');scope=rd(TR/'user_scope64.json');seal=rd(TR/'scope_hc64/sealed.json')
RESULT_PATH=ROOT/'results/state_opd_hc64/hc64.json';scores=rd(RESULT_PATH)
require(scope['queries']==seal['queries']==scores['queries']==64,'scope')
require(seal['GT_read'] is False,'pre scoring seal')
require(scores['proof']['seal_sha256']==sha(TR/'scope_hc64/sealed.json'),'scored exact seal')
source_bad=[k for k,v in lock['sources'].items() if sha(ROOT/k)!=v];require(not source_bad,'frozen sources changed')
require(sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256'],'formal state changed')
rows={x['query']:x['values'] for x in scores['queries_anonymous']}
A={a:collections.defaultdict(list) for a in ARMS};checks=collections.Counter();per={};errs=collections.defaultdict(float)
for i,key in enumerate(scope['ids'],1):
 receipt=rd(TR/'predictions'/f'{key}.json');sf=seal['files'][key]
 require(receipt['fingerprint']==lock['fingerprint'],'receipt fingerprint');require(receipt['GT_read'] is False,'receipt GT flag')
 require(sha(TR/sf['path'])==sf['sha256']==receipt['sha256'],'NPZ digest')
 require(sha(TR/'predictions'/f'{key}.json')==sf['receipt_sha256'],'receipt digest')
 require(sha(TR/receipt['trace_path'])==receipt['trace_sha256'],'trace digest')
 t=torch.load(TR/receipt['trace_path'],map_location='cpu',weights_only=False)
 with np.load(TR/sf['path'],allow_pickle=False) as npz:
  nodes=len(npz['frame_ids']);obs=np.zeros(nodes,dtype=bool);obs[npz['positions8']]=True
  E=arr(t[ARMS[0]]['path'][0]['boxes']);P=arr(t[ARMS[0]]['basis']);initial=t[ARMS[0]]['path'][0]
  require(nodes==65 and E.shape==(65,4),'full trajectory nodes')
  require(np.max(np.abs(P.T@P-np.eye(4)))<1e-10,'P orthonormal')
  query=f'H{i:03d}';per[query]={}
  for arm in ARMS:
   d=t[arm];p=d['path'];rounds=d['rounds'];c=d['costs'];ic=d['initial_checks'];x=A[arm]
   require(np.array_equal(arr(d['basis']),P),'cross arm basis');require(ic['basis_info']['uses_only_unmasked_tokens'],'unmasked basis')
   require(hashlib.sha256(arr(d['basis']).tobytes()).hexdigest()==d['basis_info']['basis_sha256'],'basis hash')
   require(np.array_equal(arr(p[0]['boxes']),E) and p[0]['U']==initial['U'] and p[0]['mask']==initial['mask'],'cross arm initial')
   require(np.count_nonzero(arr(p[0]['z']))==np.count_nonzero(arr(p[0]['eta']))==0,'zero init')
   require(ic['original_native_Source_bit_exact'] and ic['original_E_bit_exact'] and ic['network_weights_frozen'],'initial/frozen recorded')
   require(arm!='DirectFullRef' or ic['full_reference_forward_bit_exact'],'FullRef compatibility')
   require(ic['valid_mask_anchors']==8,'unexpected inactive mask')
   require(np.array_equal(arr(d['final']),npz[arm]) and np.array_equal(arr(p[-1]['boxes']),npz[arm]),'final NPZ/path agreement')
   require(len(p)==len(rounds)+1,'path count')
   require(1<=c['full_trajectory_evals']<=1321 and c['native_offset_decoder_calls']==2*c['full_trajectory_evals'],'budget')
   ev=1+sum((32 if arm in ('StateOPD','DirectMC') else 1)+len(r['backtracking']) for r in rounds)
   require(ev==c['full_trajectory_evals']==c['reward_evals'],'eval accounting')
   require(c['updates']==len(rounds) and c['accepted_updates']==sum(r['accepted'] for r in rounds),'update accounting')
   expected_back=(32*len(rounds) if arm=='DirectMC' else len(rounds) if arm!='StateOPD' else 0)
   require(c['decoder_backwards']==expected_back,'decoder backwards')
   require(c['state_only_backwards']==(len(rounds) if arm=='StateOPD' else 0),'state backwards')
   require(c['setup_trajectory_evals']==(2 if arm=='DirectFullRef' else 1),'setup count')
   U=np.array([q['U'] for q in p]);require(np.all(np.diff(U)>=-1e-14),'accepted monotonicity')
   B=np.stack([arr(q['boxes']) for q in p]);Z=np.stack([arr(q['z']) for q in p]);ETA=np.stack([arr(q['eta']) for q in p]);dpath=chartd(B,E)
   guard=.5*np.square(np.maximum(0,dpath-.25)).mean(-1)+.5*np.square(np.maximum(0,dpath-.25)).max(-1)
   er=float(np.max(np.abs(guard-np.array([q['guard'] for q in p]))));errs['guard_max_abs_error']=max(errs['guard_max_abs_error'],er)
   ue=float(np.max(np.abs(U-(np.array([q['mask'] for q in p])-.1*guard))));errs['utility_max_abs_error']=max(errs['utility_max_abs_error'],ue)
   require(er<1e-12 and ue<1e-12,'shared protected U')
   eta_norm=np.linalg.norm(ETA.reshape(len(p),-1),axis=-1);require(np.all(eta_norm<4),'state radius')
   require(np.isfinite(B).all() and np.all(B[...,2:]>0),'valid trajectory')
   z=arr(d['final_z']);eta=arr(d['final_eta']);zn=float(np.linalg.norm(z));en=float(np.linalg.norm(eta))
   # Float32 GPU basis map may differ from CPU GEMM at round-off; report rather than relax forward identity.
   eta_formula=(z@P.astype(np.float32).T)*(4/np.sqrt(np.float32(16)+np.square(z).sum()))
   errs['final_eta_formula_max_abs_error']=max(errs['final_eta_formula_max_abs_error'],float(np.max(np.abs(eta_formula-eta))))
   mo=motion(arr(d['final']),E,obs)
   rec=dict(initial_U=float(U[0]),final_U=float(U[-1]),delta_U=float(U[-1]-U[0]),initial_mask=float(p[0]['mask']),final_mask=float(p[-1]['mask']),
            final_guard=float(p[-1]['guard']),maximum_guard_on_saved_path=float(guard.max()),z_L2=zn,eta_L2=en,eta_radius_fraction=en/4,
            final_radial_mapping_scale=float(4/np.sqrt(16+zn*zn)),accepted_updates=c['accepted_updates'],updates=c['updates'],trajectory_evals=c['full_trajectory_evals'],
            stop_reason=d['stop_reason'],**mo)
   vals=rows[query]
   for base in ['ERefStrong','GeometryOPD']:
    for m in ['v','observed_actual_sIoU','unobserved_actual_sIoU']:
     rec[f'delta_{m}_pp_vs_{base}']=100*(vals[arm][m]-vals[base][m])
   per[query][arm]=rec
   for k,v in rec.items():
    if isinstance(v,(int,float)):x[k].append(v)
   x['path_guard_nonzero'].append(float((guard>0).sum()));x['path_points'].append(len(p));x['path_standardized_d_max'].append(float(dpath.max()))
   x['stopping'].append(d['stop_reason'])
   x['reward_up_v_down'].append(float(rec['delta_U']>1e-12 and rec['delta_v_pp_vs_ERefStrong']<-1e-9))
   x['reward_up_obs_up_unobs_down'].append(float(rec['delta_U']>1e-12 and rec['delta_observed_actual_sIoU_pp_vs_ERefStrong']>1e-9 and rec['delta_unobserved_actual_sIoU_pp_vs_ERefStrong']<-1e-9))
   for r in rounds:
    require(r['after']['U']>=r['before']['U']-1e-14,'round monotonic')
    if r['accepted']:require(r['after']['U']>r['before']['U']+1e-8,'accept epsilon')
    for bt in r['backtracking']:x['all_backtrack_guard'].append(bt['guard'])
    x['grad_norm'].append(r['gradient_norm']);x['Adam_delta_norm'].append(r['Adam_delta_norm'])
    if 'weights' in r:
     w=arr(r['weights']);cu=arr(r['candidate_U']);ex=np.exp((cu-cu.max())/.01);ex/=ex.sum()
     require(cu.shape==(32,) and arr(r['candidate_z']).shape==(32,8,4),'realizable joint candidate shape')
     require(np.max(np.abs(w-ex))<2e-12,'softmax objective')
     x['ESS'].append(float(1/(w*w).sum()));x['weight_max'].append(float(w.max()));x['weight_TV_to_uniform'].append(float(np.abs(w-1/32).sum()/2))
     x['candidate_U_std'].append(float(cu.std()));x['candidate_U_range'].append(float(cu.max()-cu.min()));x['candidate_U_std_over_tau'].append(float(cu.std()/.01))
     x['candidate_guards_nonzero'].append(float((arr(r['candidate_guard'])>0).sum()))
     x['candidate_guard_max'].append(float(arr(r['candidate_guard']).max()))
   checks['arm_query_checks_passed']+=1
  require(np.array_equal(arr(t['StateOPD']['rounds'][0]['candidate_z']),arr(t['DirectMC']['rounds'][0]['candidate_z'])),'same initial candidate stream')
  for ra,rb in zip(t['StateOPD']['rounds'],t['DirectMC']['rounds']):require(ra['rng_seed']==rb['rng_seed'],'shared noise stream')
 checks['queries_passed']+=1
 del t

agg={}
for arm,x in A.items():
 agg[arm]={k:stats(v) for k,v in x.items() if k!='stopping'}
 agg[arm]['stop_reasons']=dict(collections.Counter(x['stopping']))
 agg[arm]['reward_up_queries']=sum(v>1e-12 for v in x['delta_U'])
 agg[arm]['reward_up_v_down_queries']=int(sum(x['reward_up_v_down']))
 agg[arm]['reward_up_obs_up_unobs_down_queries']=int(sum(x['reward_up_obs_up_unobs_down']))
 agg[arm]['path_guard_nonzero_total']=int(sum(x['path_guard_nonzero']))
 agg[arm]['candidate_guard_nonzero_total']=int(sum(x['candidate_guards_nonzero']))
 agg[arm]['content_radius_above_90percent_queries']=sum(v>.9 for v in x['eta_radius_fraction'])
 agg[arm]['content_radius_above_99percent_queries']=sum(v>.99 for v in x['eta_radius_fraction'])
 agg[arm]['final_guard_active_queries']=sum(v>0 for v in x['final_guard'])
 agg[arm]['any_path_guard_active_queries']=sum(v>0 for v in x['maximum_guard_on_saved_path'])
 agg[arm]['query_mean_IoU_vs_E_below099']=sum(v<.99 for v in x['same_node_IoU_vs_E_mean'])
 agg[arm]['query_mean_IoU_vs_E_below095']=sum(v<.95 for v in x['same_node_IoU_vs_E_mean'])
 agg[arm]['ESS_ge31_rounds']=sum(v>=31 for v in x['ESS'])
 agg[arm]['ESS_below16_rounds']=sum(v<16 for v in x['ESS'])
 agg[arm]['largest_local_movement_three']=[dict(query=q,**per[q][arm])for q in sorted(per,key=lambda q:per[q][arm]['standardized_chart_d_max'],reverse=True)[:3]]

worst={}
for arm in ARMS:
 worst[arm]={}
 for base in ['ERefStrong','GeometryOPD']:
  selected=sorted(per,key=lambda q:per[q][arm][f'delta_v_pp_vs_{base}'])[:3]
  worst[arm][base]=[dict(query=q,**per[q][arm],old_geometry_delta_v_pp_vs_ERefStrong=100*(rows[q]['GeometryOPD']['v']-rows[q]['ERefStrong']['v']),
                       all_arm_delta_v_pp_vs_ERefStrong={a:per[q][a]['delta_v_pp_vs_ERefStrong'] for a in ARMS})for q in selected]

out=dict(status='all64_saved_trace_validity_pass',scope='HC64 exposed development, first64 fixed HC128 manifest rows',fingerprint=lock['fingerprint'],queries=64,
 GT_annotations_read=False,new_GT_scoring=False,already_scored_metrics_read=True,model_calls=0,GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
 checks=dict(checks),errors=dict(errs),frozen_source_count=len(lock['sources']),source_mismatches=source_bad,
 inputs=dict(scored_metrics_sha256=sha(RESULT_PATH),seal_sha256=sha(TR/'scope_hc64/sealed.json'),scope_sha256=sha(TR/'user_scope64.json')),
 arms=agg,worst_three_by_saved_vIoU_delta=worst,
 limits=['Real full-trajectory evaluation is verified from locked source, trace dimensions, constructor checks and exact counters; no independent decoder replay was run.',
         'Box movement uses all 65 saved physical input nodes, observed means the eight expert positions. It is not GT-support-filtered sIoU.',
         'Guard and protected U were independently recomputed from saved trajectories/mask scores; native masks were not rescored.',
         'Small output movement and near-uniform candidate weights are measured; this trial alone cannot identify whether basis, radius, decoder conditioning or reward is the primary causal restriction.',
         'The radius/map/basis were not varied; no claim of sufficient search coverage or optimal hyperparameters.',
         'FullRef changes derivative graph only. Its forward compatibility is the saved bit-exact constructor check; later paths naturally differ.'],
 seconds=time.perf_counter()-start)
(OUT/'final_validity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k]for k in ['status','queries','checks','errors','seconds']},ensure_ascii=False))
for arm,d in agg.items():
 keys=['delta_U','z_L2','eta_L2','eta_radius_fraction','corner_abs_coordinate_change_pct_image','same_node_IoU_vs_E_mean','standardized_chart_d_mean','standardized_chart_d_max','ESS','candidate_U_std_over_tau','weight_max']
 print(arm,json.dumps({k:d[k]for k in keys if k in d},ensure_ascii=False))
 print('counts',arm,{k:d[k]for k in ['reward_up_queries','reward_up_v_down_queries','reward_up_obs_up_unobs_down_queries','path_guard_nonzero_total','candidate_guard_nonzero_total','content_radius_above_90percent_queries','content_radius_above_99percent_queries','stop_reasons']})
print('local_guard_counts',json.dumps({a:{k:agg[a][k]for k in ['final_guard_active_queries','any_path_guard_active_queries','query_mean_IoU_vs_E_below099','query_mean_IoU_vs_E_below095','ESS_ge31_rounds','ESS_below16_rounds','largest_local_movement_three']}for a in ARMS},ensure_ascii=False))
