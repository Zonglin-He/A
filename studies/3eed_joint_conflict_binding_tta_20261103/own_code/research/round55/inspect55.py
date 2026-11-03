from pathlib import Path
import json,collections,numpy as np
R=Path(__file__).resolve().parents[2]
rows=json.loads((R/'research/round24/input/ALL336_PRIVATE.json').read_text());cnt=collections.Counter();fields=collections.Counter();ent=collections.Counter()
for row in rows:
 a=json.loads((R/'crossplatform/round51/grouped336_attempt001/attributes_PRIVATE'/(row['id']+'.json')).read_text());g=a['text_graph_unchanged']
 for rel in g.get('reference_relations',[]):
  cnt[(str(rel.get('frame')),str(rel.get('predicate')))]+=1;fields[tuple(sorted(rel))]+=1
  ent[(str(rel.get('target')),str(rel.get('subject')),str(rel.get('object')))]+=1
out={'n':len(rows),'relations':[{'frame':k[0],'predicate':k[1],'n':v} for k,v in cnt.items()],'fieldsets':[(list(k),v) for k,v in fields.items()],'rolekeys':[(list(k),v) for k,v in ent.items()]}
row=rows[0]
for key,path in [('latent','crossplatform/round53/latent336_attempt001/samples_PRIVATE'),('prepared','crossplatform/round51/prepared336_attempt001/arrays_PRIVATE'),('obs54','crossplatform/round54/observations336_attempt001/samples_PRIVATE')]:
 with np.load(R/path/(row['id']+'.npz'),allow_pickle=False) as z:out[key]={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
print(json.dumps(out,indent=2))
(R/'research/round55/INPUT_SCHEMA_AND_RELATION_COUNTS.json').write_text(json.dumps(out,indent=2)+'\n')
