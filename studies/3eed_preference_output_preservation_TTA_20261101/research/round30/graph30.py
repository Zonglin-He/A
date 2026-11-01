"""A third candidate can be supported by another accepted pair. Never infer negatives."""
import sys,collections,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import *
fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');byid=collections.defaultdict(list)
for f in fb:byid[f['id_PRIVATE']].append(f)
records=read(EXP/'PAIR_EXECUTION_ROWS_PRIVATE.json');lookup={r['id']:r for r in rows()};sel=read(EXP/'SELECTED_BETA_LOCK.json');out=[];stats={a:collections.Counter() for a in ['preference_only','keep_selected']}
for x in records:
 r=lookup[x['id_PRIVATE']];name='b0' if x['arm']=='preference_only' else sel['selected']['name'];p=load_npz(BASE/'screen_attempt001'/(r['platform']+'_'+name+'_A')/'predictions_PRIVATE'/(r['id']+'.npz'));q=int(p['query']);others=[f for f in byid[r['id']] if f['pair_alias']!=x['pair_alias']];supports=[f['pair_alias'] for f in others if q in f['positive_group_PRIVATE']];opposes=[f['pair_alias'] for f in others if q in f['negative_group_PRIVATE']]
 if x['final_choice_group']=='other_group_or_unassigned':
  kind='supported_by_another_accepted_pair' if supports else 'not_covered_by_any_accepted_positive';stats[x['arm']][kind]+=1
 else:kind='within_this_pair';stats[x['arm']][kind]+=1
 out.append({'arm':x['arm'],'alias':x['alias'],'pair_alias':x['pair_alias'],'platform':x['platform'],'final_choice_group':x['final_choice_group'],'third_status':kind,'other_supporting_pair_aliases':supports,'other_opposing_pair_aliases':opposes,'task_delta_vs_source_pp':x['task_delta_vs_source_pp']})
dump(EXP/'THIRD_CANDIDATE_GRAPH_ROWS.json',out);dump(EXP/'THIRD_CANDIDATE_GRAPH_SUMMARY.json',{k:dict(v) for k,v in stats.items()});print({k:dict(v) for k,v in stats.items()})
