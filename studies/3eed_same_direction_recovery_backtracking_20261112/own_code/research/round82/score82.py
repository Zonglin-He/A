"""One postseal cached-IoU join, all arms and all cases, no new GT/media reads."""
from core82 import *
import itertools
sys.path.insert(0,str(ROOT/'research/round55'));from score55 import summarize
pin(read(H82/'experiment/SCORE_CODE_LOCK.json'));P=B82/'prediction2640_attempt001';s=read(P/'SEALED.json');au=read(H82/'independent_native_audit_attempt001/AUDIT_RESULT.json');assert au['state'].startswith('PASS_') and au['prediction_seal_sha256']==sha(P/'SEALED.json');coh=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');ys={}
for path in [ROOT/'research/round55/score_dev336_attempt001',ROOT/'research/round69/score_all2304_attempt001']:
 ypath=path/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(ypath)==read(path/'SEALED.json')['files'][ypath.name];ys.update({v['id_PRIVATE']:v['candidate_IoUs_PRIVATE'] for v in read(ypath)})
statuses={v['id']:v['arm_statuses'] for v in read(P/'READOUT_STATUS_PRIVATE.json')};tasks=[]
for i,r in enumerate(allrows80()):
 f=P/'samples_PRIVATE'/(r['id']+'.npz');assert sha(f)==s['files'][f.relative_to(P).as_posix()];z=load(f);q=dict(zip(ARMS82,map(int,z['queries'])));tasks.append({'row':'X'+str(i+1).zfill(5),'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':'DEV336' if r['id'] in DEV_IDS else coh[r['id']],'split':'ALL336','queries':q,'IoU':{a:float(ys[r['id']][q[a]]) for a in ARMS82},'statuses':statuses[r['id']]})
for part,n in [('DEV336',336),('VAL1280',1280),('TRAIN1024',1024)]:
 rr=[r for r in tasks if r['part']==part];assert len(rr)==n;su,sc=summarize(rr,ARMS82,['frozen_bi','frozen_direct3','R80_raw_correct','bt_wrong','R59'],n);neg=[{'row':r['row'],'platform':r['platform'],'part':r['part'],'split':r['split'],'arm':a,'reference':b,'delta_pp':100*(r['IoU'][a]-r['IoU'][b])} for r in rr for a,b in itertools.permutations(ARMS82,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];O=H82/('score_'+part.lower()+'_attempt001');O.mkdir();receipt={'state':'PASS_R82_POSTSEAL_AUDITED_SAVED_A256_IOU_JOIN','n':n,'arm_names':ARMS82,'task_values':n*len(ARMS82),'ordered_negative_comparisons':len(neg),'single_saved_IoU_join':True,'new_GT_scored_task_values':0,'reused_saved_task_values':n*len(ARMS82),'prediction_seal_sha256':sha(P/'SEALED.json'),'no_per_input_task_GT':True}
 for k,v in [('ALL_TASK_ROWS_PRIVATE',rr),('TASK_SUMMARY',su),('ALL_SCENE_SUMMARIES_PRIVATE',sc),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SCORING_RECEIPT',receipt),('SHARED_CALIBRATION_SELECTION',None)]:dump(O/(k+'.json'),v)
 seal(O,receipt['state']);print(part,su['ALL'+str(n)]['balanced_mIoU_percent'])
