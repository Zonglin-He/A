import sys,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import *
verify(ROOT/'crossplatform/round29/direct208_attempt001');fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(fb)==22;idx={r['id']:r for r in rows()};sel=read(EXP/'SELECTED_BETA_LOCK.json');graph=read(EXP/'THIRD_CANDIDATE_GRAPH_ROWS.json');counts=collections.Counter()
for x in graph:
 r=next(r for r in rows() if r['alias']==x['alias']);name='b0' if x['arm']=='preference_only' else sel['selected']['name'];p=load_npz(BASE/'screen_attempt001'/(r['platform']+'_'+name+'_A')/'predictions_PRIVATE'/(r['id']+'.npz'));q=int(p['query']);f=next(f for f in fb if f['id_PRIVATE']==r['id'] and f['pair_alias']==x['pair_alias']);inside=q in f['positive_group_PRIVATE'] or q in f['negative_group_PRIVATE'];supports=[g['pair_alias'] for g in fb if g['id_PRIVATE']==r['id'] and g['pair_alias']!=f['pair_alias'] and q in g['positive_group_PRIVATE']];kind='within_this_pair' if inside else 'supported_by_another_accepted_pair' if supports else 'not_covered_by_any_accepted_positive';assert kind==x['third_status'];assert supports==x['other_supporting_pair_aliases'];counts['graph_pair_decisions_recomputed']+=1
records=read(EXP/'ALL336_TASK_ROWS_PRIVATE.json');expected=[]
for r in records:
 for arm in r['q']:
  refs=['source','preference_only']+(['same_feedback_direct'] if r['split']=='A208' else ['all_A_zeroU_memory'] if r['split']=='S64' else [])
  for ref in refs:
   if r['ious'][arm]<r['ious'][ref]-1e-10:expected.append((r['alias'],r['split'],arm,ref,100*(r['ious'][arm]-r['ious'][ref])))
actual=[(r['alias'],r['split'],r['arm'],r['reference'],r['delta_pp']) for r in read(EXP/'ALL_NEGATIVES.json')];assert sorted(actual)==sorted(expected)
labels=read(BASE/'evidence_review22_attempt001/ANNOTATIONS.json');key=read(EXP.parent/'input/EVIDENCE22_KEY_PRIVATE.json');assert len(labels)==len(key)==22;types=collections.Counter(x['available_evidence_type'] for x in labels);assert dict(types)==read(EXP/'EVIDENCE22_TYPE_COUNTS.json')['counts'];assert 'geometry_relation_only' not in types
s=read(EXP/'CPU_AUDIT.json');assert s['state']=='PASSED';s.update(graph_audit_counts=dict(counts),all_negative_comparisons_retained=len(actual),unmodified_original_feedback_seal_verified=True,evidence_counts_independently_reduced=dict(types));dump(EXP/'CPU_AUDIT.json',s);print('GRAPH_AND_NEGATIVES_AUDIT_PASSED',len(actual))
