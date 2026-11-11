"""Status-only correction; identical already-scored predictions, no new fit."""
from common72 import *
import os
old=ROOT/'crossplatform/round72/predicted2640_attempt002';out=ROOT/'crossplatform/round72/predicted2640_attempt003';assert not out.exists();out.mkdir();(out/'samples_PRIVATE').mkdir();p=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');e=ROOT/p['evidence_root'];ad=ROOT/p['adapt_root'];base={x['id']:x['arm_statuses'] for x in read(ROOT/'crossplatform/round71/predicted2640_attempt001/READOUT_STATUS_PRIVATE.json')};stats=[];changed=0
previous={x['id']:x['arm_statuses'] for x in read(old/'READOUT_STATUS_PRIVATE.json')}
for r in rows():
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';os.link(old/rel,out/rel);v=load(e/rel);fit=read(ad/'cases_PRIVATE'/(sid+'.json'));st={}
 for name in ARMS:
  if name in ARMS[:4]:st[name]=base[sid]['CLIP_direct' if name=='CLIP_reference' else name].copy();continue
  key='RGB_N' if name.startswith('RGBshuffle') else 'Nshuffle_N' if name.startswith('Nshuffle') else 'box_N' if name.startswith('box_') else 'color_N' if name.startswith('color_') else 'relation_N' if name.startswith('relation_') else 'main_N';N=v[key][v['query_group']];eligible=bool(N.any() and not N.all());tag='RGBshuffle' if name.startswith('RGBshuffle') else 'Nshuffle' if name.startswith('Nshuffle') else 'graph';up='_tta' in name;active=fit[tag]['active'] if up else eligible
  known=v['target_box_known'] if key=='box_N' else v['target_known'];st[name]={'unknown':not bool(known.any()),'fallback':not active,'no_parameter_update':not(up and active)}
 changed+=sum(st[k]!=previous[sid][k] for k in ARMS);stats.append({'id':sid,'arm_statuses':st})
dump(out/'READOUT_STATUS_PRIVATE.json',stats);cost=read(old/'COSTS.json');cost.update(status_fields_repaired=changed,all_prediction_NPZ_bytes_identical=True);dump(out/'COSTS.json',cost);seal(out,'R72_STATUS_ONLY_REPAIR_IDENTICAL_SEALED_PREDICTION_BYTES',n=2640,arm_names=ARMS);dump(H/'STATUS_REPAIR_RECEIPT.json',{'state':'PASS_R72_STATUS_ONLY_NO_PREDICTION_CHANGES','n':2640,'corrected_status_records':changed,'all_prediction_NPZ_bytes_identical':True,'original_prediction_seal_sha256':sha(old/'SEALED.json'),'corrected_prediction_seal_sha256':sha(out/'SEALED.json'),'new_model_fits':0,'new_task_predictions':0});print(read(H/'STATUS_REPAIR_RECEIPT.json'))
