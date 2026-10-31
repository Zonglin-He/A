"""Descriptive post-seal failure localization; no alternative method output."""
from pathlib import Path
import sys,collections,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round16'))
from common import EXP,read,dump,sha,verify
def main():
 inp=EXP.parent/'input';cases={c['id']:c for c in read(inp/'A_CASES_PRIVATE.json')};old=ROOT/'crossplatform/round16/G_A32_attempt002';local=ROOT/'crossplatform/round16/G_A32_attempt004';verify(old);verify(local)
 rows=[];summary={}
 for r in read(inp/'A32_PRIVATE.json'):
  c=cases[r['id']];g=read(old/'cases_PRIVATE'/(r['id']+'.json'));l=read(local/'cases_PRIVATE'/(r['id']+'.json'))['preference'];ev=g['preference']['all_mask_evidence'];assignment=np.array(l['query_mask_assignment']);mapped=np.unique(assignment[assignment>=0])
  with np.load(ROOT/c['mask_dir']/(c['frame_alias']+'.npz'),allow_pickle=False) as z:classes=z['class_ids'].copy()
  f=np.array([math.log(max(e['full']['probabilities']['YES'],1e-12)/max(e['full']['probabilities']['NO'],1e-12)) for e in ev]);w=np.array([math.log(max(e['wrong_caption']['probabilities']['YES'],1e-12)/max(e['wrong_caption']['probabilities']['NO'],1e-12)) for e in ev]);pairs=[(int(m),int(n)) for m in mapped for n in mapped if m!=n and classes[m]==classes[n]]
  full=[(m,n) for m,n in pairs if f[m]-f[n]>=math.log(2)];rev=[(m,n) for m,n in full if w[n]-w[m]>=math.log(2)];loc=[(m,n) for m,n in rev if ev[m]['location']['status']!='contradict']
  row={'alias':g['alias'],'platform':r['platform'],'mask_count':len(ev),'mapped_mask_count':len(mapped),'same_class_mapped_ordered_pairs':len(pairs),'full_margin_pairs':len(full),'reversal_pairs':len(rev),'location_compatible_pairs':len(loc),'final_usable':l['status']=='usable','full_pair_abs_gap_max':max([abs(f[m]-f[n]) for m,n in pairs],default=None),'full_wrong_logodds_change_max':float(np.max(np.abs(f-w))) if len(f) else None,'source_query_mapped_to_mask':bool(assignment[int(np.load(ROOT/'crossplatform/round16/features64_attempt001/samples_PRIVATE'/(r['id']+'.npz'))['scores'].argmax())]>=0)}
  rows.append(row)
 for p in ('drone','quad'):
  pp=[r for r in rows if r['platform']==p];summary[p]={'n':len(pp),'mask_count':sum(r['mask_count'] for r in pp),'no_mapped_masks':sum(r['mapped_mask_count']==0 for r in pp),'at_least_two_mapped_masks':sum(r['mapped_mask_count']>=2 for r in pp),'has_same_class_mapped_pair':sum(r['same_class_mapped_ordered_pairs']>0 for r in pp),'has_full_margin_pair':sum(r['full_margin_pairs']>0 for r in pp),'has_caption_reversal_pair':sum(r['reversal_pairs']>0 for r in pp),'has_location_compatible_pair':sum(r['location_compatible_pairs']>0 for r in pp),'final_usable':sum(r['final_usable'] for r in pp),'source_query_mapped':sum(r['source_query_mapped_to_mask'] for r in pp),'maximum_full_sameclass_pair_gap':max([r['full_pair_abs_gap_max'] for r in pp if r['full_pair_abs_gap_max'] is not None],default=None),'maximum_caption_logodds_change':max(r['full_wrong_logodds_change_max'] for r in pp if r['full_wrong_logodds_change_max'] is not None)}
 dump(EXP/'FAILURE_STAGE_DIAGNOSIS.json',{'state':'POSTSCORE_DESCRIPTIVE_NO_NEW_PREDICTIONS_OR_TUNING','program_sha256':sha(__file__),'summary':summary,'anonymous_rows':rows,'not_independent_query_or_mask_trials':True,'same_scene_wrong_caption_may_describe_same_physical_object':True,'caption_reversal_is_stronger_than_generic_relevance_not_a_universal_TTA_requirement':True});print(summary)
if __name__=='__main__':main()
