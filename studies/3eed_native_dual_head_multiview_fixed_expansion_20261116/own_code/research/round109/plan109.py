"""Metadata-only, scene-stratified non-DEV expansion; no labels/model/media decode."""
from pathlib import Path
import json,hashlib,datetime,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round109';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not (H/'experiment/INPUT_LOCK.json').exists();frames=rd(R/'research/round105/FRAMES_PRIVATE.json');ix={x['metapath']:x for x in frames};summary=[];anchors=[]
 for pl in ['waymo','drone','quad']:
  dev=rd(R/f'research/round99/input/TARGET_{pl.upper()}128_PRIVATE.json');seen={str(Path(x['metapath']).resolve()) for x in dev};rows=rd(R/f'research/round97/input/TARGET_{pl.upper()}_PRIVATE.json');groups=collections.defaultdict(list)
  for r in rows:
   mp=str(Path(r['metapath']).resolve())
   if mp not in seen:groups[r['scene']].append(r)
  order=sorted(groups,key=lambda s:hashlib.sha256(('R109scene|'+pl+'|'+s).encode()).hexdigest())
  for sc in order:groups[sc].sort(key=lambda r:hashlib.sha256(('R109row|'+r['id']).encode()).hexdigest())
  selected=[];used=set()
  while len(selected)<256:
   moved=False
   for sc in order:
    while groups[sc] and str(Path(groups[sc][0]['metapath']).resolve()) in used:groups[sc].pop(0)
    if groups[sc]:
     r=groups[sc].pop(0);mp=str(Path(r['metapath']).resolve());assert mp in ix;used.add(mp);selected.append(r);anchors.append({'A':ix[mp],'B_pair':None,'C_pair':None,'status':'planned' if ix[mp]['eligible_for_extra_media'] else 'protected_content_alias_frozen_fallback'});moved=True
     if len(selected)==256:break
   assert moved,'not enough distinct nonDEV physical paths; do not silently resample'
  dump(H/f'input/TARGET_{pl.upper()}256_PRIVATE.json',selected);summary.append({'target':pl,'expressions':len(selected),'distinct_anchor_metapaths':len(used),'scene_clusters':len({r['scene'] for r in selected}),'protected_alias_fallback':sum(not ix[str(Path(r['metapath']).resolve())]['eligible_for_extra_media'] for r in selected),'current_DEV_anchor_metapath_overlap':0})
 dump(H/'ANCHORS_STAGE0_PRIVATE.json',anchors);dump(H/'INPUT_SUMMARY.json',{'groups':summary,'unique_target_expressions':768,'directed_inputs':1536,'no_model_GT_or_media_decode':True,'historically_exposed_full_VAL_not_independent_test':True});dump(H/'experiment/INPUT_LOCK.json',{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selector':'fixed R109 scene hash roundrobin then row hash; distinct meta paths; excludes current DEV anchor paths, never label/score selected','target_rows':{p.name:sha(p) for p in sorted((H/'input').glob('*.json'))},'anchor_sha256':sha(H/'ANCHORS_STAGE0_PRIVATE.json'),'protected_alias_policy':'include in denominator, no fresh decode/model, allnewarms cached prepared fallback; original_source reference retained','protected_registry':'same R105 registered frame inventory, excludes registered FINAL inputs, flags protected RGB content aliases','method':'exact R106 native real/randomA/wrong LiDAR fusion; exact R108 CE/BBF equal pools; no parameter/statistics updates or tuning','primary':'real_softtoken_geometric_fusedABC','controls':['original_source','prepared_frozen','real_BBF_fusedABC','real_CE_fusedABC','real_softtoken_arithmetic_fusedABC','randomA_same_budget_native_BBF_and_dualheads','wrong_same_values_binding_native_BBF_and_dualheads','CE_queryshuffle_geometric'],'no_experts':True,'source_retraining':False,'FINAL_access':False,'all_failures_unknowns_fallbacks_retained':True,'deadline_utc':'2026-11-17T01:00:00+00:00'});print(json.dumps(rd(H/'INPUT_SUMMARY.json'),indent=2))
if __name__=='__main__':main()
