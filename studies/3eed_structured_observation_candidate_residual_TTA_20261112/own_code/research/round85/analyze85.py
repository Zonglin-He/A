"""Saved-results aggregation; no method tuning or candidate filtering."""
from core85 import *
import collections
P=B85/'prediction366_attempt002';rows=read(H85/'experiment/ALL366_PRIVATE.json');tasks=read(H85/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json')+read(H85/'score_supp30_attempt001/ALL_TASK_ROWS_PRIVATE.json');tm={x['id_PRIVATE']:x for x in tasks};coverage=[];updates=[];lines=[]
for i,r in enumerate(rows):
 x=read(P/'records_PRIVATE'/(r['id']+'.json'));sample=load(P/'samples_PRIVATE'/(r['id']+'.npz'));coverage.append({'row':'CASE'+str(i+1).zfill(4),'platform':r['platform'],'part':'DEV336' if r['id'] in DEV_IDS else 'SUPP30','active_evidence':x['active_evidence'],'cost_range':x['cost_range'],'cost_at_source_top1':x['cost_at_source_top1'],'direct_changes_source_query':int(sample['queries'][ARMS85.index('direct_constraints')])!=int(sample['queries'][0]),'relation_terms':len(x['relations']),'max_zero_residual_error':x['zero_step_max_error']})
 for v,trajectory in x['variants'].items():
  for st in trajectory:updates.append({'row':'CASE'+str(i+1).zfill(4),'platform':r['platform'],'part':'DEV336' if r['id'] in DEV_IDS else 'SUPP30','variant':v,**{k:val for k,val in st.items() if k!='trials'}})
dump(H85/'EVIDENCE_COVERAGE_ROWS.json',coverage);dump(H85/'ALL_UPDATE_ROWS.json',updates);methods=[];scenes=[]
for part in ['dev336','supp30']:
 folder=H85/('score_'+part+'_attempt001');su=read(folder/'TASK_SUMMARY.json');name='ALL336' if part=='dev336' else 'ALL30';s=su[name]
 for a in ARMS85:
  methods.append({'cohort':part.upper(),'platform':'balanced','arm':a,'mIoU_percent':s['balanced_mIoU_percent'][a],'gain_vs_source_pp':s['balanced_mIoU_percent'][a]-s['balanced_mIoU_percent']['source'],'gain_vs_direct_pp':s['balanced_mIoU_percent'][a]-s['balanced_mIoU_percent']['direct_constraints'],'gain_vs_R59_pp':s['balanced_mIoU_percent'][a]-s['balanced_mIoU_percent']['R59']})
  for pl in ['drone','quad']:methods.append({'cohort':part.upper(),'platform':pl,'arm':a,**s[pl][a]})
 sr=read(folder/'ALL_SCENE_SUMMARIES_PRIVATE.json');vals={pl:sorted({x['scene_PRIVATE'] for x in sr if x['platform']==pl}) for pl in ['drone','quad']}
 for x in sr:
  for a in ARMS85:scenes.append({'cohort':part.upper(),'platform':x['platform'],'scene_alias':x['platform'][0].upper()+'_S'+str(vals[x['platform']].index(x['scene_PRIVATE'])+1),'n':x['n'],'arm':a,'mIoU_percent':x['mIoU_percent'][a]})
dump(H85/'ALL_METHOD_ROWS.json',methods);dump(H85/'ALL_SCENE_ROWS.json',scenes)
caseout=[];local=[]
for i,c in enumerate(read(ROOT/'research/round84/experiment/CASES33_PRIVATE.json')):
 sid=c['row']['id'];x=tm[sid];record=read(P/'records_PRIVATE'/(sid+'.json'));old=read(ROOT/'crossplatform/round84/capture33_attempt001'/(c['case']+'_PRIVATE.json'))['chosen_PRIVATE'];a={'case':'CASE'+str(i+1).zfill(3),'part':x['part'],'platform':x['platform'],'active_evidence':record['active_evidence'],'source_IoU':x['IoU']['source'],'R76_frozen_IoU':x['IoU']['frozen_bi'],'R80_raw_IoU':c['task']['IoU']['R80_raw_correct'],'R82_BT_IoU':x['IoU']['R82_BT'],'direct_IoU':x['IoU']['direct_constraints'],'full1_IoU':x['IoU']['full1'],'full3_IoU':x['IoU']['full3'],'full1_matches_source_candidate':x['queries']['full1']==old['source'],'full1_matches_R80_raw_candidate':x['queries']['full1']==old['raw'],'full1_matches_R82_BT_candidate':x['queries']['full1']==old['BT'],'new_identity_not_automatically_certified':True};caseout.append(a);local.append(dict(a,case_PRIVATE=c['case']))
dump(H85/'R84_CASE_FOLLOWUP_ROWS.json',caseout);dump(H85/'R84_CASE_FOLLOWUP_PRIVATE.json',local)
# Target-proxy coverage is postseal diagnosis only, never an optimization input.
ys={}
for fold in ['research/round55/score_dev336_attempt001','research/round69/score_all2304_attempt001']:ys.update({x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/fold/'ALL_CANDIDATE_IOUS_PRIVATE.json')})
target=[]
for i,r in enumerate(rows):
 sid=r['id'];z0=prefix(r) if sid in DEV_IDS else {'xyz':load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),keys=['A_xyz'])['A_xyz']};a=load(P/'samples_PRIVATE'/(sid+'.npz'));boxes=a['source_boxes'];q=int(np.argmax(ys[sid]));inside=(abs(z0['xyz'][0]-boxes[q,:3])<=boxes[q,3:]/2).all(-1);g=load(casepath(r) if sid in DEV_IDS else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz'));region={'point':a['region_point'],'image':a['region_image']};v=[]
 for protected in [False,True]:
  mm=[]
  for k in [85001,85002]:
   rg=np.random.default_rng(int(hashlib.sha256(('R85localmask|'+sid+'|'+str(k)).encode()).hexdigest()[:16],16));remaining=np.flatnonzero(g['depth_valid']);n=len(remaining)//4;order=[]
   for count in [n,n]:
    chosen=rg.permutation(remaining[region['image'][remaining]])[:int(np.ceil(.75*count))];other=rg.permutation(remaining[~np.isin(remaining,chosen)])[:count-len(chosen)];take=np.r_[chosen,other];order.append(take);remaining=remaining[~np.isin(remaining,take)]
   pm=g['visible']&np.isin(g['cell'],order[1])
   if protected:pm[g['query_anchor_indices']]=False
   mm.append(pm)
  v.append(int(((mm[0]|mm[1])&inside).sum()))
 target.append({'row':'CASE'+str(i+1).zfill(4),'part':'DEV336' if r['id'] in DEV_IDS else 'SUPP30','active_evidence':bool(np.ptp(a['cost'])>1e-8),'oracle_proxy_point_targets_unprotected':v[0],'oracle_proxy_point_targets_protected':v[1],'offline_GT_proxy_only':True})
dump(H85/'POSTSEAL_TARGET_PROXY_COVERAGE.json',target)
for n in ['ALL_METHOD_ROWS','EVIDENCE_COVERAGE_ROWS','ALL_UPDATE_ROWS','ALL_SCENE_ROWS','R84_CASE_FOLLOWUP_ROWS','POSTSEAL_TARGET_PROXY_COVERAGE']:
 assert (H85/(n+'.json')).exists()
print([(x['arm'],x['mIoU_percent']) for x in methods if x['cohort']=='DEV336' and x['platform']=='balanced'])
