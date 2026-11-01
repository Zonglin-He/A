import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round29'))
from common29 import *
labels=[('X','high','White sedan between white car and silver SUV.'),('X','high','Right person has light upper clothing; left companion is dark.'),('Y','high','White sedan; X is silver SUV.'),('X','high','Light-shirt right person; other person ahead-left.'),('U','unknown','Overlapping car boxes and partly occluded dark sedan; cannot certify central mask identity from box panel.'),('X','high','Brown SUV in road next to bus; Y is white distant car.'),('Y','medium','Dark sedan behind front SUV and next to blue car, partially occluded.'),('Y','high','Dark sedan behind white vehicle; X is white vehicle.'),('Y','high','Gray roof-rack SUV beside white sedan.'),('N','high','Described cropped dark car is at lower right, neither marked vehicle.'),('Y','medium','White sedan at right next to black foreground car; overlapping box limits confidence.'),('X','medium','White roof-rack wagon upper left, Y nearby sedan.'),('X','high','Left black SUV next to another SUV on right.'),('Y','high','Foreground text-shirt skateboarder, X seated reference person.'),('Y','high','Light-shirt backpack pedestrian; X is dark person ahead.'),('N','high','Described rear-facing dark SUV at far right is unmarked; X white SUV and Y brown sedan.'),('Y','high','Crouching dark-top person behind standing dark-trouser person.'),('X','high','Light-shirt person left, dark person behind-right.'),('Y','high','Brown sedan with white SUV to right and white sedan left.'),('X','high','Foreground dark long-sleeve skateboarder; Y seated reference.'),('Y','high','Light-shirt left pedestrian; X dark reference.'),('X','high','Left black SUV, second dark SUV right.'),('X','medium','Partly occluded background light-shirt skateboarder at right; foreground torso matches coarse attributes too.'),('Y','high','Left light-shirt person front facing; X backpack figure ahead-right.')]
out=BASE/'blind_review24_attempt001';out.mkdir(exist_ok=False);dump(out/'ANNOTATIONS.json',[{'case':f'C{i+1:02d}','target':l,'confidence':c,'reason':s} for i,(l,c,s) in enumerate(labels)]);dump(out/'SCOPE.json',{'annotator':'assistant, single visual review','not_expert_gold':True,'scores_GT_and_source_choice_hidden_during_review':True,'N':'neither candidate; not arbitrary missing annotation','U':'unresolved, retained','medium_labels_are_provisional':True,'review_preference_not_used_as_runtime_pseudo_label':True});seal(out,'R29_blind_identity_annotations_before_unblinding',n=24)
# Only after annotation seal, join existing numeric feedback.
pr={(x['id_PRIVATE'],x['pair_alias']):x for x in read(EXP/'PAIR154_ROWS_PRIVATE.json')};rec=[]
for k,a in zip(read(EXP.parent/'input/BLIND24_KEY_PRIVATE.json'),read(out/'ANNOTATIONS.json')):
 x=pr[k['id_PRIVATE'],k['pair_alias']];target=next((z for z in ['A','B'] if a['target'] in k['labels_to_mask_PRIVATE'] and k['labels_to_mask_PRIVATE'][a['target']]==k['mask_'+z+'_PRIVATE']),a['target']);rec.append({'case':a['case'],'alias':k['alias'],'platform':k['platform'],'target_AB':target,'confidence':a['confidence'],'policies':x['policies'],'iou_A':x['iou_A'],'iou_B':x['iou_B']})
dump(EXP/'BLIND24_UNBLINDED_ROWS.json',rec)
import collections,json
summary={}
for name in rec[0]['policies']:
 cc=collections.Counter()
 for x in rec:
  t=x['target_AB'];p=x['policies'][name]
  if t=='U':cc['unresolved']+=1;continue
  if p=='U':cc['abstained']+=1;continue
  cc['correct' if t==p else 'wrong']+=1
  if p=='B':cc['B_correct' if t=='B' else 'B_wrong']+=1
 summary[name]=dict(cc)
dump(EXP/'BLIND24_SUMMARY.json',summary);print(json.dumps(summary,indent=2))
