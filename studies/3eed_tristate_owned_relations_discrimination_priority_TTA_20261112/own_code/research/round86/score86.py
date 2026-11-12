"""Single postseal existing-IoU join. Main DEV336 and supplements kept separate."""
from core86 import *
import itertools
sys.path.insert(0,str(ROOT/'research/round55'));from score55 import summarize
P=B86/'prediction366_attempt001';ss=read(P/'SEALED.json');audit=read(H86/'independent_native_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(P/'SEALED.json');ys={}
loss_audit=read(H86/'independent_loss_replay_audit_attempt001/AUDIT_RESULT.json');assert loss_audit['state'].startswith('PASS_') and loss_audit['prediction_seal_sha256']==sha(P/'SEALED.json')
for folder in ['research/round55/score_dev336_attempt001','research/round69/score_all2304_attempt001']:
 f=ROOT/folder/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(f)==read(ROOT/folder/'SEALED.json')['files'][f.name];ys.update({x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(f)})
statuses={r['id']:r['arm_statuses'] for r in read(P/'READOUT_STATUS_PRIVATE.json')};tasks=[]
for i,r in enumerate(read(H86/'experiment/ALL366_PRIVATE.json')):
 f=P/'samples_PRIVATE'/(r['id']+'.npz');assert sha(f)==ss['files'][f.relative_to(P).as_posix()];a=load(f);q=dict(zip(ARMS86,map(int,a['queries'])));tasks.append({'row':'X'+str(i+1).zfill(5),'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':'DEV336' if r['id'] in DEV_IDS else 'SUPP30','split':'ALL336','queries':q,'IoU':{k:float(ys[r['id']][v]) for k,v in q.items()},'statuses':statuses[r['id']]})
for part,n in [('DEV336',336),('SUPP30',30)]:
 rows=[x for x in tasks if x['part']==part];assert len(rows)==n;su,sc=summarize(rows,ARMS86,['source','frozen_bi','R82_BT','R59','R85_full1','F_direct_guard','F_candidate1_guard','F_mismatch1_guard','F_sum1_guard','F_joint1_raw'],n);neg=[{'row':r['row'],'platform':r['platform'],'part':r['part'],'split':r['split'],'arm':a,'reference':b,'delta_pp':100*(r['IoU'][a]-r['IoU'][b])} for r in rows for a,b in itertools.permutations(ARMS86,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];out=H86/('score_'+part.lower()+'_attempt001');out.mkdir();receipt={'state':'PASS_R86_POSTSEAL_SAVED_A256_IOU_JOIN','n':n,'arm_names':ARMS86,'task_values':n*len(ARMS86),'ordered_negative_comparisons':len(neg),'single_saved_IoU_join':True,'new_GT_scored_task_values':0,'reused_saved_task_values':n*len(ARMS86),'prediction_seal_sha256':sha(P/'SEALED.json'),'no_task_GT_in_evidence_or_optimization':True}
 for key,value in [('ALL_TASK_ROWS_PRIVATE',rows),('TASK_SUMMARY',su),('ALL_SCENE_SUMMARIES_PRIVATE',sc),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SCORING_RECEIPT',receipt),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(key+'.json'),value)
 seal(out,receipt['state']);print(part,su['ALL'+str(n)]['balanced_mIoU_percent'],flush=True)
