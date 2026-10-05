"""CPU only: check complete-query orders, per-video roster and unit boundaries."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
def run():
 d=read(BASE/'DESIGN_LOCK.json');m=read(BASE/'MASTER_PLAN.json');checks=[]
 def check(name,x):assert x,name;checks.append(name)
 for f,h in {**d['inputs'],**d['protected']}.items():check('hash:'+f,sha(ROOT/f)==h)
 check('canonical-frozen',m['canonical_method']==read(ROOT/'methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json'))
 total=0
 for ds,n,nv,ns in [('vidstg',10303,732,732),('hc2',3482,3482,237)]:
  p=read(BASE/ds/'PLAN.json');r=read(BASE/ds/'ONE_QUERY_VIDEO_ROSTER.json');rows=p['rows']
  check(ds+':query-complete',len(rows)==p['queries']==n)
  check(ds+':video-count',len({x['input']['original_video_id'] for x in rows})==nv)
  check(ds+':cluster-count',len({x['source'] for x in rows})==ns)
  check(ds+':three-orders',len(p['orders'])==3)
  for o,seq in p['orders'].items():
   check(ds+':'+o+':permutation',sorted(seq)==list(range(n)))
   seen=set();prev=None
   for i in seq:
    source=rows[i]['source']
    if source!=prev:check(ds+':'+o+':block:'+str(len(seen)),source not in seen);seen.add(source);prev=source
   check(ds+':'+o+':seal',digest(seq)==m['order_hashes'][ds][o])
  selected=r['parent_ordinals'];videos=collections.defaultdict(list)
  for x in rows:videos[x['input']['original_video_id']].append(x)
  expected={min(xs,key=lambda x:(digest(['PaperOneQueryVideo-20261005',ds,x['key'],x['input']['caption']]),x['ordinal']))['ordinal'] for xs in videos.values()}
  check(ds+':hash-selected',len(selected)==nv and set(selected)==expected)
  check(ds+':roster-order',r['order1']==[i for i in p['orders']['order1'] if i in expected])
  check(ds+':selection-label-free',not r['GT_used'] and r['selection_before_predictions'])
  check(ds+':clean-only',p['conditions']==['clean'])
  total+=n*3
 check('table1:ours41355',total==d['online_arrivals']==m['stages'][0]['ours_arrivals']==41355)
 check('table2:67424',16*(732+3482)==m['stages'][1]['arrivals_per_online_method'])
 check('old:paused-no-partial-scoring',read(ROOT/'artifacts/decota_fixed_full_corruption_v1/STATUS.json')['status']=='paused_by_user_revised_paper_scope' and not m['old_partial_scored'])
 out=dict(status='pass',checks=len(checks),details=checks,GPU_seconds=0,GT_read=False,time=time.time())
 write(BASE/'DESIGN_ROOT_ACCEPTANCE.json',out);print('DESIGN_ROOT_PASS',len(checks),flush=True)
if __name__=='__main__':run()
