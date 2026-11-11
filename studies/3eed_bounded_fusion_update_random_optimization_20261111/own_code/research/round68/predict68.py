"""Five gamma, four actual eta heads, three random repeats; no combinations."""
import sys,time,torch
from common68 import *
from fusion68 import cpu_head,parts,fusion
sys.path.insert(0,str(ROOT/'research/round58'));import numerics58 as n
H=ROOT/'research/round68';p=read(H/'experiment/PREDICT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);assert read(ROOT/p['scaled_root']/'SEALED.json')['n']==336;initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');base=cpu_head(initial);out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();torch.set_grad_enabled(False);torch.set_num_threads(4);start=time.perf_counter();statuses=[]
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';ab=load(ROOT/p['AB_root']/rel);c=load(ROOT/p['C_root']/rel);g=load(ROOT/p['association_root']/rel);prior=load(ROOT/p['r59_root']/rel);r66=load(ROOT/p['r66_prediction_root']/rel);r67=load(ROOT/p['r67_prediction_root']/rel);random=load(ROOT/p['random_capture_root']/rel);old=load(ROOT/p['r67_capture_root']/rel);records=read(ROOT/p['scaled_root']/'cases_PRIVATE'/(sid+'.json'));statefile=ROOT/p['scaled_root']/rel;scaled=load(statefile) if statefile.exists() else {};hs={v:(c['h'] if v=='C' else ab[v+'_h']) for v in 'ABC'};z={v+'_'+k:(c[k] if v=='C' else ab[v+'_'+k]) for v in 'ABC' for k in ['tokens','token_mask']};d=n.prepared(ab,c,g);pr=parts(base,hs,z,g);result={'source':prior['scores'][prior['arm_names'].tolist().index('source')],'R59':prior['scores'][prior['arm_names'].tolist().index('direct_all')],'R66':r66['scores'][r66['arm_names'].tolist().index('R59_tta1')],'C_source':n.distribution(base,d,'C').numpy()};meta={}
 for ga in GAMMAS:result['G_'+tag(ga)]=fusion(pr,z,g,ga)
 for ga,key in [(0.,'frozen_ABC'),(1.,'direct_all')]:assert np.max(abs(result['G_'+tag(ga)]-prior['scores'][prior['arm_names'].tolist().index(key)]))<1e-6 and result['G_'+tag(ga)].argmax()==prior['scores'][prior['arm_names'].tolist().index(key)].argmax()
 for eta in ETAS:
  key=tag(eta);arr={k:scaled[key+'_'+k] for k in initial} if records[key]['eligible'] else initial;m=cpu_head(arr);assert n.phash(m)==records[key]['state_sha256'];result['E_'+key]=fusion(parts(m,hs,z,g),z,g,1.);result['C_E_'+key]=n.distribution(m,d,'C').numpy();meta[key]=records[key]
 assert np.max(abs(result['E_1p0']-result['R66']))<1e-11 and result['E_1p0'].argmax()==result['R66'].argmax()
 result['random_s0']=r67['scores'][r67['arm_names'].tolist().index('random_R59')]
 for rep in [1,2]:
  hz={v:random['s'+str(rep)+'_'+v+'_h'] for v in 'ABC'};zz={v+'_tokens':random['s'+str(rep)+'_'+v+'_tokens'] for v in 'ABC'};zz.update({v+'_token_mask':old[v+'_token_mask'] for v in 'ABC'});result['random_s'+str(rep)]=fusion(parts(base,hz,zz,g),zz,g,1.)
 scores=np.stack([result[a] for a in ARMS]);q=scores.argmax(1);assert np.isfinite(scores).all();np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=ab['A_boxes'],source_scores=ab['A_scores'],selected_original_boxes=ab['A_boxes'][q]);dump(out/'cases_PRIVATE'/(sid+'.json'),meta);st={}
 for a in ARMS:
  update=(a.startswith('E_') or a.startswith('C_E_'));k=a.split('E_')[-1] if update else None;st[a]={'unknown':False,'fallback':bool(update and records[k]['eligible'] and not records[k]['accepted']),'no_parameter_update':not (update and records[k]['parameter_delta_L2']>0)}
 st['R66']=dict(st['E_1p0']);statuses.append({'id':sid,'arm_statuses':st})
 if(i+1)%64==0:print('PREDICT68',i+1,flush=True)
dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',{'n':336,'arms':len(ARMS),'seconds':time.perf_counter()-start,'C_after_scaled_state_seal':True});seal(out,'R68_ALL_TWELVE_CONFIGURATIONS_AND_C_DIAGNOSTICS_SEALED_BEFORE_SCORE',n=336,arm_names=ARMS);print({'n':336,'arms':len(ARMS)})
