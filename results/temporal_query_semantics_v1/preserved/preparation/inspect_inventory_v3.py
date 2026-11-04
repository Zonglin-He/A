from pathlib import Path
import json,collections,hashlib,itertools
R=Path.cwd();D=R/'outputs/temporal_query_semantics_v1';read=lambda p:json.loads(p.read_text())
rr=[json.loads(x)for x in(R/'data/evaluation/vidstg.jsonl').read_text().splitlines()];ann=read(R/'data/annotations/vidstg_tastvg_eval.json');cat=read(R/'data/annotations/vidstg_test.json');lookup=collections.defaultdict(list)
for n,v in enumerate(cat):
 for form in ['captions','questions']:
  for z in v[form]:lookup[(str(v['vid']),z['description'],z['target_id'],v['used_segment']['begin_fid'],v['used_segment']['end_fid']+1,v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid'])].append((n,v,z,form))
joined=[];dup=[];ambiguous=[]
for r in rr:
 m=r['metadata'];a=ann[str(m['index'])];assert a['sentence']['description']==m['caption']and str(a['vid'])==str(m['original_video_id']);z=a['sentence'];iv=a['ori_temp_gt'];k=(str(a['vid']),z['description'],z['target_id'],m['start_frame'],m['end_frame'],iv['begin_fid'],iv['end_fid']);ls=lookup[k];assert len(ls)>0,r['id'];rel={json.dumps(x[1]['used_relation'],sort_keys=True)for x in ls};
 if len(rel)!=1:ambiguous.append(dict(id=r['id'],possible_relations=sorted(rel)));continue
 if len(ls)>1:dup.append(r['id'])
 n,v,z,form=ls[0];joined.append(dict(id=r['id'],index=m['index'],video=str(v['vid']),video_sha=m['video_sha256'],segment=[m['start_frame'],m['end_frame']],frame_ids=m['frame_ids'],target=z['target_id'],caption=z['description'],form=form,catalog_index=n,relation=v['used_relation'],event=[iv['begin_fid'],iv['end_fid']],category=next(o['category']for o in v['subject/objects']if o['tid']==z['target_id'])))
static={'above','behind','beneath','in_front_of','inside','left','right','next_to','towards','away'}
g=collections.defaultdict(list)
for x in joined:
 if x['form']=='captions'and x['target']==x['relation']['subject_tid']and x['relation']['predicate']not in static:g[(x['video'],x['target'],x['relation']['object_tid'],tuple(x['frame_ids']))].append(x)
def ti(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))/(max(a[1],b[1])-min(a[0],b[0]))
pairs=[]
for v in g.values():
 for a,b in itertools.combinations(v,2):
  if a['relation']['predicate']==b['relation']['predicate']or ti(a['event'],b['event'])>.2:continue
  if a['id']>b['id']:a,b=b,a
  h=hashlib.sha256(f'20261104|{a["video"]}|{a["target"]}|{a["id"]}|{b["id"]}'.encode()).hexdigest();pairs.append(dict(a=a,b=b,order_sha256=h,GT_event_tIoU=ti(a['event'],b['event'])))
pairs.sort(key=lambda x:x['order_sha256']);chosen=[];videos=set()
for x in pairs:
 if x['a']['video']not in videos:chosen.append(x);videos.add(x['a']['video'])
 if len(chosen)==32:break
summary=dict(total=10303,joined=len(joined),duplicate_catalog_records_identical_relation=len(dup),ambiguous_relation_queries_excluded=len(ambiguous),eligible_pairs=len(pairs),eligible_videos=len({x['a']['video']for x in pairs}),selected_pairs=len(chosen),selected_unique_queries=len({x[s]['id']for x in chosen for s in ['a','b']}),selection='up to32independentvideos, oneSHA-firstpair/video, sameoriginaltrackactor+sameobjecttrack+exactframegrid+differentnonstaticpredicate+GTeventtIoU<=.2; before model outcome access',predicates=sorted({x[s]['relation']['predicate']for x in pairs for s in ['a','b']}))
(D/'inventory_v3.json').write_text(json.dumps(dict(summary=summary,selected=chosen,eligible=pairs,ambiguous_catalog_relations=ambiguous,excluded_static_predicates=sorted(static)),ensure_ascii=False,indent=2));print(json.dumps(summary))
for i,x in enumerate(chosen):
 print('PAIR',i,x['a']['video'],x['GT_event_tIoU'])
 for s in ['a','b']:
  a=x[s];print(a['id'],a['event'],a['relation']['predicate'],a['caption'])
