"""Before task scoring, replay engineering states on CPU to validate saved schema."""
from audit125 import *
def main():
 torch.set_num_threads(4);cols=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');out=[]
 for src in ['waymo','drone','quad']:
  S=dict(np.load(R/cols[src]['state_path']));head={k[6:]:v for k,v in S.items() if k.startswith('match.')}
  for co in ['DEV','EXP']:
   for e in reg:
    if e['source']!=src or e['cohort']!=co or e['protected_alias_fallback']:continue
    D=B/'engineering'/co/(src+'_to_'+e['target']);f=D/'records_PRIVATE'/(e['row']['id']+'.json')
    if f.exists() and rd(f)['fit']['active']:out.append({'source':src,'cohort':co,**validate(e,D,head,S)});break
 assert len(out)==6;(H/'CPU_PROBE.json').write_text(json.dumps({'state':'PASS6_REAL_TEACHER_CONSENSUS_STATE_FREE_FIELD_HEAD_AND_OBJECTIVE_SCHEMA','n':len(out),'checks':out,'GT_used':False},indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
