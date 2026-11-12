"""Postseal fixed-arm diagnostics; no parameter or gate selection."""
from core86 import *
P=B86/'prediction366_attempt001';rows=read(H86/'experiment/ALL366_PRIVATE.json');tasks=read(H86/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json')+read(H86/'score_supp30_attempt001/ALL_TASK_ROWS_PRIVATE.json');tm={r['id_PRIVATE']:r for r in tasks};coverage=[];updates=[];methods=[];scenes=[];targets=[]
ys={}
for f in ['research/round55/score_dev336_attempt001','research/round69/score_all2304_attempt001']:ys.update({x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/f/'ALL_CANDIDATE_IOUS_PRIVATE.json')})
for nr,r in enumerate(rows):
 sid=r['id'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));x=read(P/'records_PRIVATE'/(sid+'.json'));part='DEV336' if sid in DEV_IDS else 'SUPP30';tag='CASE'+str(nr+1).zfill(4);ev=load(B86/'evidence366_attempt002/samples_PRIVATE'/(sid+'.npz'))
 for b,rec in x['bases'].items():
  p=np.exp(a[b+'_s0']-np.max(a[b+'_s0']));p/=p.sum();base=int(p.argmax());pi=a[b+'_pi'];raw=int(pi.argmax());guard=int(guarded86(pi,a[b+'_eligible']).argmax());known=a[b+'_known'];labels=ev['labels'];logs=rec['factor_logs'];t=tm[sid]
  coverage.append({'row':tag,'part':part,'platform':r['platform'],'base':b,'active':rec['active'],'factor_count':len(logs),'known_queries':rec['known_queries'],'eligible_queries':rec['eligible_queries'],'unknown_probability_error':rec['unknown_probability_error'],'base_top_known':bool(known[base]),'base_top_has_counter':any(z['base_group_has_counter'] for z in logs),'raw_changes_base':raw!=base,'guard_changes_base':guard!=base,'raw_winner_unknown':not bool(known[raw]),'raw_winner_ineligible':not bool(a[b+'_eligible'][raw]),'raw_and_guard_differ':raw!=guard,'direct_raw_IoU':t['IoU'][b+'_direct_raw'],'direct_guard_IoU':t['IoU'][b+'_direct_guard']})
  for v,trajectory in rec['variants'].items():
   for st in trajectory:updates.append({'row':tag,'part':part,'platform':r['platform'],'base':b,'variant':v,**{k:v for k,v in st.items() if k!='trials'}})
 z0=prefix(r) if sid in DEV_IDS else {'xyz':load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),keys=['A_xyz'])['A_xyz']};q=int(np.argmax(ys[sid]));inside=(abs(z0['xyz'][0]-a['source_boxes'][q,:3])<=a['source_boxes'][q,3:]/2).all(-1);old=next(v for v in read(ROOT/'research/round85/POSTSEAL_TARGET_PROXY_COVERAGE.json') if v['row']==tag)
 targets.append({'row':tag,'part':part,'platform':r['platform'],'new_point_targets_in_oracle_proxy':int((a['mask_point'].any(0)&inside).sum()),'R85_unprotected_proxy_targets':old['oracle_proxy_point_targets_unprotected'],'R85_protected_proxy_targets':old['oracle_proxy_point_targets_protected'],'offline_GT_proxy_only':True})
for part,n in [('dev336',336),('supp30',30)]:
 su=read(H86/('score_'+part+'_attempt001/TASK_SUMMARY.json'))['ALL'+str(n)]
 for arm in ARMS86:
  methods.append({'cohort':part.upper(),'platform':'balanced','arm':arm,'mIoU_percent':su['balanced_mIoU_percent'][arm],'gain_vs_source_pp':su['balanced_mIoU_percent'][arm]-su['balanced_mIoU_percent']['source'],'gain_vs_frozen_bi_pp':su['balanced_mIoU_percent'][arm]-su['balanced_mIoU_percent']['frozen_bi'],'gain_vs_F_direct_guard_pp':su['balanced_mIoU_percent'][arm]-su['balanced_mIoU_percent']['F_direct_guard']})
  for pl in ['drone','quad']:methods.append({'cohort':part.upper(),'platform':pl,'arm':arm,**su[pl][arm]})
 sr=read(H86/('score_'+part+'_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json'));vals={pl:sorted({x['scene_PRIVATE'] for x in sr if x['platform']==pl}) for pl in ['drone','quad']}
 for x in sr:
  for arm in ARMS86:scenes.append({'cohort':part.upper(),'platform':x['platform'],'scene_alias':x['platform'][0].upper()+'_S'+str(vals[x['platform']].index(x['scene_PRIVATE'])+1),'n':x['n'],'arm':arm,'mIoU_percent':x['mIoU_percent'][arm]})
caseout=[]
for i,c in enumerate(read(ROOT/'research/round84/experiment/CASES33_PRIVATE.json')):
 t=tm[c['row']['id']];out={'case':'CASE'+str(i+1).zfill(3),'part':t['part'],'platform':t['platform']}
 for arm in ARMS86:out[arm+'_IoU']=t['IoU'][arm]
 out['new_identity_not_automatically_certified']=True;caseout.append(out)
for name,obj in [('EVIDENCE_COVERAGE_ROWS',coverage),('ALL_UPDATE_ROWS',updates),('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS',scenes),('POSTSEAL_TARGET_PROXY_COVERAGE',targets),('R84_CASE_FOLLOWUP_ROWS',caseout)]:dump(H86/(name+'.json'),obj)
print([(x['arm'],x['mIoU_percent']) for x in methods if x['cohort']=='DEV336' and x['platform']=='balanced'])
