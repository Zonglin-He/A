"""One-query/source, hash-only cohort and exact specialist demand. No labels or scores."""
import sys,time,hashlib,collections,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_common_v1 import BASE,OLD,CONDS,FAMILIES,read,write,sha

def h(s):return hashlib.sha256(s.encode()).hexdigest()
def run():
 b=read(OLD/'ROSTER_LOCK.json');groups=collections.defaultdict(list)
 for r in b['rows']:groups[r['source']].append(r)
 rows=[]
 for source in sorted(groups):
  r=min(groups[source],key=lambda r:(h('P48-query-v1|'+r['key']),r['key']))
  rows.append({**r,'b1_ordinal':r['ordinal'],'ordinal':len(rows)})
 assert len(rows)==len({r['source'] for r in rows})==670
 subset=sorted(range(670),key=lambda i:(h('P48-subset-v1|'+rows[i]['source']),rows[i]['source']))
 orders={f'order{j}':sorted(range(670),key=lambda i:(h(f'P48-order-v1|{j}|'+rows[i]['source']),rows[i]['source'])) for j in [1,2]}
 demands=collections.defaultdict(set);panels={}
 for name,ids,conds,availability in [('P1',set(range(670)),CONDS,25),('P2',set(subset[:128]),['clean']+[f'{f}_{s}' for f in FAMILIES for s in [1,5,10]],25)]+[(f'P3_b{x}',set(subset[:64]),CONDS,x) for x in [0,25,100]]:
  order={k:[i for i in seq if i in ids] for k,seq in orders.items()};total=len(ids)*2*len(conds)
  plan=dict(name=name,rows=rows,orders=order,conditions=conds,availability=availability,total=total,sources=len(ids),GT_used=False,globally_fresh=False,source_checkpoint='Vid-trained TA-STVG frozen J01',same_video_repeated_within_stream=False)
  write(BASE/name/'PLAN.json',plan);panels[name]=dict(sources=len(ids),arrivals=total,conditions=conds,availability=availability)
  for seq in order.values():
   for pos,i in enumerate(seq):
    if availability==100 or (availability==25 and pos%4==0):demands[i].update(conds)
  # Alias subject parses by selected query; original receipt hash verified first.
  sb=read(OLD/'SUBJECT_BARRIER.json');files={}
  for i,r in enumerate(rows):
   src=OLD/'subjects'/f"{r['b1_ordinal']:05}.json";assert sha(src)==sb['files'][src.name]
   dst=BASE/name/'subjects'/f'{i:05}.json';dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst);files[dst.name]=sha(dst)
  write(BASE/name/'SUBJECT_BARRIER.json',dict(count=len(rows),files=files,original_barrier_sha256=sha(OLD/'SUBJECT_BARRIER.json')))
 write(BASE/'EXPERT_PLAN.json',dict(rows=rows,conditions=['clean']+[f'{f}_{s}' for f in FAMILIES for s in [1,5,10]],expert_needed=sorted(demands),conditions_by_parent={str(i):sorted(c) for i,c in demands.items()},total=sum(map(len,demands.values())),GT_used=False))
 write(BASE/'COHORT.json',dict(status='locked_before_new_panel_predictions',sources=670,queries=670,query_rule='min SHA256(P48-query-v1|key)',order_rule='SHA256(P48-order-v1|j|source)',subset_rule='SHA256(P48-subset-v1|source)',panels=panels,GT_used_for_selection=False,globally_fresh=False,parent_sha256=sha(OLD/'ROSTER_LOCK.json'),time=time.time()))
 write(BASE/'ORDERS.json',{name:{k:[f'S{i+1:04}' for i in seq] for k,seq in read(BASE/name/'PLAN.json')['orders'].items()} for name in panels})
 write(BASE/'P4_PLAN.json',dict(parents=subset[:100],condition='clean',availability=25,uncached=True,total=100,GT_used=False))
 print('LOCKED',panels,'expert pairs',sum(map(len,demands.values())),flush=True)
if __name__=='__main__':run()
