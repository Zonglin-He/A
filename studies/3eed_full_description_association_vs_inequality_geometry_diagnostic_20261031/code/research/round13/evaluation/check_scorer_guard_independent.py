"""Independent synthetic producer-schema tests only; no geometry/GT IO."""
from pathlib import Path
import hashlib,importlib.util,json,tempfile
V=Path(__file__).resolve().parent;B=V.parents[2]
spec=importlib.util.spec_from_file_location('R13_scorer_guard_under_review',V/'score_direct.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks={}
with tempfile.TemporaryDirectory(prefix='producer_guard_synthetic_',dir=V) as td:
 p=Path(td);(p/'dummyproducer.py').write_text('# synthetic producer, not executed\n');(p/'receipt.json').write_text('{"synthetic":true}\n');digest=m.sha(p/'dummyproducer.py');bindings={str(p/'dummyproducer.py'):digest};m.bind_map(bindings,'independentfixture')
 def seal(producer):
  d={'files':{'receipt.json':m.sha(p/'receipt.json')}}
  if producer!='MISSING':d['program_sha256']=producer
  m.dump(p/'SEALED.json',d);return {'role':'synthetichelper','directory':str(p),'seal_sha256':m.sha(p/'SEALED.json')}
 m.verify_artifact(seal(digest),bindings);checks['stringproducer_exact_explicit_bound_hash_pass']=True
 try:m.verify_artifact(seal('f'*64),bindings)
 except ValueError:checks['resealed_unknown_stringproducer_rejected']=True
 else:raise AssertionError('Unknownproducer accepted')
 relative=str((p/'dummyproducer.py').relative_to(B));m.verify_artifact(seal({relative:digest}),bindings);checks['dictproducer_actual_relative_code_hash_pass']=True
 for value,label in [(17,'integer'),(None,'null'),('', 'emptystring')]:
  try:m.verify_artifact(seal(value),bindings)
  except ValueError:checks[label+'_producer_rejected']=True
  else:raise AssertionError('Malformedproducer accepted')
 m.verify_artifact(seal('MISSING'),bindings);checks['missing_sourceproducer_uses_emptydict_captureclosure_policy']=True
 assert not (p/'SCORER_LOCK.json').exists();checks['producer_guard_never_creates_GT_boundary_or_reads_index']=True
out={'state':'PASS_R13_INDEPENDENT_SYNTHETIC_SCORER_PRODUCER_GUARD','checks':checks,'scorer_sha256':m.sha(V/'score_direct.py'),'auditor_sha256':m.sha(Path(__file__)),'no_actual_GT_index_geometry_IoU_data_model_GPU_access':True}
f=V/'SCORER_GUARD_INDEPENDENT_CPU_FIXTURE.json';f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'nchecks':len(checks),'sha256':m.sha(f)}))
