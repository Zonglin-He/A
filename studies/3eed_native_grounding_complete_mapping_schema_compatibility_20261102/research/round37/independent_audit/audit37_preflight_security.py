"""Small independent preflight selection and inert strict parser contract checks."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from audit37_strict_independent import parse,rows,read,dump

def main():
 rr=rows('A208');chosen=[]
 def key(r):return hashlib.sha256(('R37_preflight_v1|'+json.dumps({k:v for k,v in r.items() if k not in ['alias','stage','frame_alias']},sort_keys=True,separators=(',',':'))).encode()).hexdigest()
 for pl in ['drone','quad']:
  pools={s:sorted([r for r in rr if r['platform']==pl and r['scene']==s],key=key) for s in sorted({r['scene'] for r in rr if r['platform']==pl})};selected=[]
  while len(selected)<6:
   for sc in pools:
    if pools[sc] and len(selected)<6:selected.append(pools[sc].pop(0))
  chosen.extend(selected)
 actual=read(ROOT/'research/round37/experiment/PREFLIGHT12_PRIVATE.json');assert [r['id'] for r in chosen]==[r['id_PRIVATE'] for r in actual]
 bad=['__import__("os").system("echo BAD")','{"point":[1,NaN]}','{"point":[1,2],"point":[2,3]}','{"point":[true,2]}','{"point":"1,2"}','{"point":[1,2],"label":"x"}','[{"point":[1,2]}]']
 assert all(parse(t,'point')['status']=='parse_failure' for t in bad)
 good=[('{"point":[1,2]}','parsed'),('```json\n{"point2d":null}\n```','explicit_null'),('<answer>{"point_2d":[1.5,2]}</answer>','parsed')]
 assert all(parse(t,'point')['status']==s for t,s in good);assert parse('{"target_id":true}','som')['status']=='parse_failure';assert parse('{"target_id":1}','som')['status']=='parsed'
 result={'state':'PASS','preflight_inputhash_scene_roundrobin_12_exact':True,'strict_safe_JSON_rejection_cases':len(bad)+1,'strict_wrappers_aliases_null_checks':len(good)+1,'generated_text_never_executed':True,'singleton_list_is_correctly_rejected_in_original_R37':True,'this_does_not_assert_rejection_is_semantically_required':True,'no_GT_GPU_newVLM_optimizer_FINAL_identity_access':True};dump(HERE/'PREFLIGHT_SECURITY_AUDIT.json',result);print(result)
if __name__=='__main__':main()
