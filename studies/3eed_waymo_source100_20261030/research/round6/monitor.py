"""Read-only finite training progress snapshot; no model/GPU/GT/scoring action."""
from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parent;B=R.parents[1];RUN=B/'crossplatform/round6/waymo_source100';log=B/'crossplatform/round6/formal100.log'
def lines(path):
 if not path.exists():return []
 out=[]
 for l in path.read_text(errors='replace').splitlines():
  try:v=json.loads(l)
  except (json.JSONDecodeError,ValueError):continue
  if isinstance(v,dict):out.append(v)
 return out

def main():
 events=[]
 for f in sorted((RUN/'attempts').glob('*/events.jsonl')):events.extend(lines(f))
 if not events:print(json.dumps({'state':'starting'}));return
 complete=[x for x in events if x.get('event')=='epoch_training_complete'];latest=events[-1];done=complete[-1]['epoch'] if complete else 0;progress=[x for x in events if x.get('event')=='train_progress'];p=progress[-1] if progress else {};vals=lines(RUN/'MILESTONES.jsonl');end=(RUN/'TRAINING_COMPLETE.json').exists();fails=list((RUN/'attempts').glob('*/INTERRUPTED_OR_FAILED.json'));scalar={'state':'completed' if end else ('failed' if fails else 'running'),'epoch_training_completed':done,'latest_epoch':p.get('epoch'),'latest_batch1based':p.get('batch',-1)+1,'latest_global_step':p.get('global_step',0),'latest_step_loss':p.get('loss'),'last_completed_epoch_mean_loss':complete[-1].get('mean_loss') if complete else None,'last_epoch_training_seconds':complete[-1].get('train_seconds') if complete else None,'last_event':latest.get('event'),'elapsed_attempt_seconds':round(latest['elapsed_seconds'],1),'completed_source_val_epochs':[x['epoch'] for x in vals],'peak_allocated_bytes_so_far':max([x.get('peak_allocated_bytes',0) for x in progress],default=0)}
 if complete:
  recent=statistics.median(x['train_seconds'] for x in complete[-5:]);val=statistics.median(x['source_val_seconds'] for x in vals) if vals else 35.;scalar['rough_remaining_minutes_from_completed_epochs'] = round(((recent+1.5)*(100-done)+val*((100//10)-(done//10)))/60,1);scalar['ETA_scope']='recent epoch wall including worker startup +1.5s save + observed sourceVAL or35s placeholder; not guarantee'
 if '--brief' in __import__('sys').argv:scalar={k:scalar[k] for k in ['state','epoch_training_completed','latest_epoch','latest_batch1based','latest_global_step','last_completed_epoch_mean_loss','completed_source_val_epochs','rough_remaining_minutes_from_completed_epochs'] if k in scalar}
 print(json.dumps(scalar,allow_nan=False))
if __name__=='__main__':main()
