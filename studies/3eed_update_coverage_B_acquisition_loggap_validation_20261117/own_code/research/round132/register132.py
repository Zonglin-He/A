"""Original registration math, added counters only; stop at first pass among fixed B candidates."""
from common132 import *
import sys,time,concurrent.futures,multiprocessing,collections
sys.path.insert(0,str(R132/'research/round131'));import register131 as G
def measured(pair):
 log=[];current=[];original_visible=G.backend._visible_raw;original_matches=G.backend.rgb_matches
 def visible(*a,**kw):
  x=original_visible(*a,**kw);current.append(len(x[0]));return x
 def matches(*a,**kw):
  x=original_matches(*a,**kw);log.append({'visible_depth_A':current[-2],'visible_depth_B':current[-1],'RGB_matches':len(x[0])});return x
 G.backend._visible_raw=visible;G.backend.rgb_matches=matches
 try:r=G.worker(pair)
 finally:G.backend._visible_raw=original_visible;G.backend.rgb_matches=original_matches
 r['instrumentation_visible_call_counts']=current;r['instrumentation_forward_reverse']=log;r['instrumentation_changes_computation']=False;return r
def work(job):
 kind,a=job;attempts=[];chosen=None;total=0.
 cc=a['candidates'] if kind=='target' else ([{'rank':1,'pair':a,'registration_reuse':None}])
 for c in cc:
  pair=c['pair'];reuse=c['registration_reuse'];pid=pair['pair_PRIVATE']
  if reuse:r=rd(R132/reuse);path=reuse;cost=0.
  else:
   r=measured(pair);path=str((B132/'registration_PRIVATE'/(pid+'.json')).relative_to(R132));dump(R132/path,r);cost=r['seconds'];total+=cost
  attempts.append({'rank':c['rank'],'pair_PRIVATE':pid,'path':path,'sha256':sha(R132/path),'geometry_cycle_supported':r['geometry_cycle_supported'],'status':r['status'],'reused':bool(reuse),'new_seconds':cost})
  if r['geometry_cycle_supported']:chosen=c['rank'];break
 result={'kind':kind,'A_frame_PRIVATE':(a['A'] if kind=='target' else a['A'])['frame_PRIVATE'],'selected_rank':chosen,'selected_pair_PRIVATE':attempts[-1]['pair_PRIVATE'] if chosen else None,'selected_registration':attempts[-1]['path'] if chosen else None,'attempts':attempts,'new_registration_seconds':total,'no_GT_or_semantic_or_score_selection':True};key=('TARGET_' if kind=='target' else 'SOURCE_')+result['A_frame_PRIVATE'];dump(B132/'selections_PRIVATE'/(key+'.json'),result);return result
def main():
 basecheck();plan=rd(H132/'experiment/DESIGN_LOCK.json')
 for n,h in plan['files'].items():assert sha(H132/n)==h
 assert not (B132/'registration_PRIVATE').exists();(B132/'registration_PRIVATE').mkdir();(B132/'selections_PRIVATE').mkdir();jobs=[('target',a) for a in rd(H132/'TARGET_B3_CANDIDATES_PRIVATE.json')]+[('source',p) for p in rd(H132/'SOURCE_PAIRS_PRIVATE.json')];start=time.perf_counter();out=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for i,r in enumerate(pool.map(work,jobs,chunksize=1)):
   out.append(r)
   if (i+1)%32==0:print('REGISTER132',i+1,len(jobs),round(time.perf_counter()-start,1),flush=True)
 summary={'jobs':len(out),'seconds':time.perf_counter()-start,'groups':{kind:{'n':sum(r['kind']==kind for r in out),'selected_ranks':dict(collections.Counter(str(r['selected_rank']) for r in out if r['kind']==kind)),'new_attempts':sum(not a['reused'] for r in out if r['kind']==kind for a in r['attempts']),'reused_attempts':sum(a['reused'] for r in out if r['kind']==kind for a in r['attempts']),'sum_new_pair_seconds':sum(r['new_registration_seconds'] for r in out if r['kind']==kind)} for kind in ['target','source']},'gates_unchanged':True,'no_GT_or_model_used':True,'FINAL_access':False};dump(H132/'REGISTRATION_SUMMARY.json',summary);dump(B132/'REGISTRATION_SEALED.json',{'files':{str(p.relative_to(B132)):sha(p) for folder in ['registration_PRIVATE','selections_PRIVATE'] for p in (B132/folder).glob('*.json')},'jobs':len(out),'plan_sha256':sha(H132/'experiment/DESIGN_LOCK.json')});print(summary)
if __name__=='__main__':main()
