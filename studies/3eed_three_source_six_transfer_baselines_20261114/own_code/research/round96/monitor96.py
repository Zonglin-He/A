"""Read-only progress; no model, annotations, CUDA, prediction or scoring access."""
from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';rd=lambda p:json.loads(p.read_text());out={'asof_local':datetime.datetime.now().astimezone().isoformat(),'state':rd(H/'STATE.json'),'sources':{}}
for pl in ['drone','quad']:
 d=B/(pl+'_source100');x={'exists':d.exists(),'completed':(d/'TRAINING_COMPLETE.json').exists()}
 if (d/'PROGRESS.json').exists():x['last_complete_epoch']=rd(d/'PROGRESS.json')
 attempts=sorted((d/'attempts').glob('*')) if d.exists() else []
 if attempts:
  ev=attempts[-1]/'events.jsonl'
  if ev.exists():
   lines=ev.read_text().splitlines();events=[]
   for l in lines:
    try:events.append(json.loads(l))
    except json.JSONDecodeError:pass
   if events:x['latest_event']={k:v for k,v in events[-1].items() if k not in ['sampler_order_sha256','first_batch_anno_indices']}
  st=attempts[-1]/'steps.jsonl'
  if st.exists():
   try:x['latest_step']=json.loads(st.read_text().splitlines()[-1])
   except (json.JSONDecodeError,IndexError):pass
 out['sources'][pl]=x
print(json.dumps(out,ensure_ascii=False,indent=2))
