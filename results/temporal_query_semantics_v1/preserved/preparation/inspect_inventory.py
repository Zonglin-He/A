from pathlib import Path
import json,collections,hashlib,re,time
R=Path.cwd();D=R/'outputs/temporal_query_semantics_v1'
read=lambda p:json.loads(p.read_text())
records=[json.loads(x)for x in(R/'data/evaluation/vidstg.jsonl').read_text().splitlines()];catalog=read(R/'data/annotations/vidstg_test.json');lookup=collections.defaultdict(list)
for n,v in enumerate(catalog):
 for form in ['captions','questions']:
  for z in v[form]:lookup[(str(v['vid']),z['description'],int(v['used_segment']['begin_fid']),int(v['used_segment']['end_fid'])+1)].append((n,v,z,form))
joined=[];miss=[]
for r in records:
 m=r['metadata'];key=(str(m['original_video_id']),m['caption'],m['start_frame'],m['end_frame']);ls=lookup[key]
 if len(ls)!=1:miss.append(dict(id=r['id'],matches=len(ls)));continue
 n,v,z,form=ls[0];joined.append(dict(id=r['id'],video=str(v['vid']),video_sha=m['video_sha256'],segment=[m['start_frame'],m['end_frame']],frame_ids=m['frame_ids'],target=z['target_id'],target_type=z['type'],caption=z['description'],form=form,catalog_index=n,relation=v['used_relation'],event=[v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid']],category=next(o['category']for o in v['subject/objects']if o['tid']==z['target_id'])))
allgroups=collections.defaultdict(list);exactgroups=collections.defaultdict(list)
for r in joined:
 if r['form']!='captions':continue
 allgroups[(r['video'],r['target'])].append(r)
 exactgroups[(r['video'],r['target'],tuple(r['frame_ids']))].append(r)
unique=lambda g:len({tuple(x['event'])for x in g})>1
agg=dict(total_queries=len(records),joined=len(joined),ambiguous_or_missing=miss,same_target_multi_event_groups=sum(unique(x)for x in allgroups.values()),same_target_same_input_multi_event_groups=sum(unique(x)for x in exactgroups.values()),same_target_same_input_multi_event_queries=sum(len(x)for x in exactgroups.values()if unique(x)))
(D/'inventory.json').write_text(json.dumps(dict(summary=agg,groups=[g for g in exactgroups.values()if unique(g)],all_target_groups=[g for g in allgroups.values()if unique(g)]),ensure_ascii=False,indent=2));print(json.dumps(agg))
for g in [g for g in exactgroups.values()if unique(g)][:35]:
 print('GROUP',g[0]['video'],g[0]['target'],g[0]['segment'],len(g))
 for x in g[:8]:print(x['id'],x['event'],x['relation']['predicate'],x['caption'])
