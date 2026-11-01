"""Same manually reviewed instances, existing visual evidence only, no new inference."""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
def main():
 keys=read(EXP.parent/'input/BLIND_KEY_PRIVATE.json');ann={x['case']:x for x in read(BASE/'blind_review_attempt001/ANNOTATIONS.json')};lookup={x['id']:x for x in rows()};out=[];counts={}
 methods=['crop_scene_joint','scene_base','scene_hires','scene_repeat','R26_scene','old_subject_raw_dual','old_subject_gray_dual']
 for key in keys:
  r=lookup[key['id_PRIVATE']];ev=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.npz')) if False else read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'));old=read(paths(r)['evidence']/'cases_PRIVATE'/(r['id']+'.json'));a=ann[key['blind_case']];vals={}
  for n in methods:
   v={}
   for label,m in key['labels_to_mask_PRIVATE'].items():
    if n.startswith('old_subject'):
     x=old['targets_PRIVATE'][str(m)];v[label]=float(np.mean([y['full']['log_odds_yes_no']-(y['gray']['log_odds_yes_no'] if n=='old_subject_gray_dual' else 0) for y in x]))
    else:v[label]=ev['targets_PRIVATE'][str(m)][n]['log_odds_yes_no']
   top=[l for l,x in v.items() if abs(x-max(v.values()))<=1e-7];targets=set(a['target_labels']);status='visual_unresolved' if not targets else 'correct' if set(top)<=targets else 'incorrect' if not set(top)&targets else 'tie_target_other';vals[n]={'status':status,'top':top,'scores':v}
   counts.setdefault(n,collections.Counter());counts[n][status]+=1
  out.append({'case':key['blind_case'],'platform':r['platform'],'methods':vals})
 dump(EXP/'VISUAL_SAME_INSTANCES_ROWS.json',out);dump(EXP/'VISUAL_SAME_INSTANCES_SUMMARY.json',{n:dict(c) for n,c in counts.items()});print('VISUAL_COMPARE28',{n:dict(c) for n,c in counts.items()})
if __name__=='__main__':main()
