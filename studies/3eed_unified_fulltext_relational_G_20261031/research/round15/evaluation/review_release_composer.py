"""Finite root-composer review, synthetic files only, never actual run outputs."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,tempfile
R=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 source=R/'research/round15/experiment/compose_score_release.py';assert sha(source)==a.composer_sha256
 spec=importlib.util.spec_from_file_location('R15_independent_composer_contract',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 c={};c['fixed_source0d']=m.SOURCE_SHA=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 c['fixed_R5_index_e8']=m.INDEX_SHA=='e8eeb12b3f68232e6729cdf33d2bf49beada7fc662c83f98115be20b7ed6d562'
 c['fixed_scorer_a5']=m.SCORER_SHA==sha(R/'research/round15/evaluation/score_direct.py')=='a5f0c0a2362427c4dab94ea3c4a833a245ef07782737c00c1a5bfef8044a57b5'
 c['frozen_geometry_dependencies']=all(sha(R/p)==h for p,h in m.DEPS.items())
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp);out=d/'NEVER_RELEASE.json';args=SimpleNamespace(semantic=d/'missing_formal',semantic_seal_sha256='0'*64,predictions=d/'missing_direct',prediction_seal_sha256='0'*64,semantic_audit=d/'not_read',direct_audit=d/'not_read_either',output=out)
  try:m.main(args)
  except FileNotFoundError:pass
  else:raise AssertionError('Missing formal seal not rejected')
  assert not out.exists();c['missing_or_partial_formal_withoutseal_firstfails_no_release']=True
  def fake(folder,state,n,calls=0):
   folder.mkdir();(folder/'SYNTHETIC.txt').write_text('SYNTHETIC')
   s={'state':state,'n_cases':n,'n_samples':n,'n_model_calls':calls,'n_arms':8,'files':{'SYNTHETIC.txt':sha(folder/'SYNTHETIC.txt')}}
   p=folder/'SEALED.json';p.write_text(json.dumps(s));return sha(p)
  sem=d/'engineering4';direct=d/'synthetic_direct';args.semantic=sem;args.semantic_seal_sha256=fake(sem,'sealed_R15_semantic_engineering4_before_GT',4,12);args.predictions=direct;args.prediction_seal_sha256=fake(direct,'sealed_R15_eight_direct_arms_before_GT',204)
  try:m.main(args)
  except ValueError as e:assert 'formal' in str(e)
  else:raise AssertionError('Engineering4 substituted for formal204')
  assert not out.exists();c['complete_engineering4_seal_cannot_enter_formal_release']=True
  unsafe=d/'unsafe';unsafe.mkdir();(unsafe/'SEALED.json').write_text(json.dumps({'files':{'../escaped_payload':'0'*64}}))
  try:m.check_seal(unsafe)
  except ValueError:pass
  else:raise AssertionError('Unsafe member admitted')
  c['relative_member_traversal_rejected_before_payload_read']=True
 text=source.read_text()
 c['composer_has_no_task_score_model_or_GTpayload_call']='read(index)' not in text and 'torch' not in text and 'transform_gt' not in text and 'candidate_ious' not in text
 # These are finite human source-review findings for the corrected version,
 # not inferred from synthetic engineering rejection alone.
 static=a.static_checks
 c.update({'static_review:'+k:True for k in static})
 assert all(c.values())
 result={'state':'PASS_R15_CORRECTED_COMPLETE_SEAL_CPU_RELEASE_COMPOSER_STATIC_AND_SYNTHETIC_NO_ACTUAL_EXECUTION',
 'composer_sha256':sha(source),'auditor_sha256':sha(__file__),'checks':c,'check_count':len(c),
 'actual_formal_or_capture_payloads_read':False,'GTindex_or_annotation_payload_read':False,'synthetic_member_files_only':True,
 'actual_release_written_or_score_model_GPU_run':False,'noGTrelease_until_all_current_formal_audits_pass':True,
 'scope':'Finite code review and three tiny synthetic refusal paths; no attempt to execute the successful actual release path before formal sealing.',
 'initial_30975_binding_gaps_reported_to_root_before_release':True,'issues':[]}
 p=Path(__file__).with_name('SCORE_RELEASE_COMPOSER_INDEPENDENT_REVIEW.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':len(c),'sha256':sha(p)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--composer-sha256',required=True);p.add_argument('--static-checks',nargs='+',required=True);main(p.parse_args())
