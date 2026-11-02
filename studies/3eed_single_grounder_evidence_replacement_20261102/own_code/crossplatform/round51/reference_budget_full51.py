"""UNEXECUTED pure-text full5912 reference-call registration, CPU only.

Counts every reference in the already sealed literal graph collection. No raw
media/metadata, source prediction arrays, GT, tokenizer or model is opened.
This budget does not authorize capture or imply that a method was adopted.
"""
from pathlib import Path
import argparse,hashlib,json,re,time

ROOT=Path(__file__).resolve().parents[2]
MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
TEXT='crossplatform/round50/full5912_attempt001/text'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
 temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def path(value):
 p=(ROOT/value).resolve();assert p.is_relative_to(ROOT) and 'FINAL' not in str(p) and 'round46/final' not in str(p)
 return p


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True)
 args=parser.parse_args();pp=path(args.plan);plan=read(pp);grant=read(path(args.authorization))
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_FULL5912_PURE_TEXT_REFERENCE_BUDGET_NO_GT'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_FULL5912_PURE_TEXT_REFERENCE_BUDGET' and plan['n']==5912
 assert plan['manifest']==MANIFEST and plan['pure_text_root']==TEXT
 assert plan['no_GT_media_source_prediction_or_models'] is True
 assert set(plan['inputs_sha256'])=={MANIFEST,TEXT+'/SEALED.json'}
 assert plan['programs_sha256']['crossplatform/round51/reference_budget_full51.py']==sha(__file__)
 for group in ('inputs_sha256','programs_sha256'):
  for rel,digest in plan[group].items():assert sha(path(rel))==digest,rel
 rows=read(ROOT/MANIFEST);seal=read(ROOT/TEXT/'SEALED.json')
 assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 assert {p:sum(r['platform']==p for r in rows) for p in ('drone','quad')}=={'drone':2984,'quad':2928}
 assert seal['state']=='R50_FULL5912_LITERAL_TEXT_COMPLETE' and seal['n']==5912
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir()
 work=[];total=0;byplatform={'drone':0,'quad':0};hist={};start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];assert re.fullmatch('[A-Za-z0-9_-]+',sid)
   rel='cases_PRIVATE/'+sid+'.json';p=ROOT/TEXT/rel;digest=sha(p);assert digest==seal['files'][rel]
   graph=read(p);refs=graph['references']
   assert len(refs)<=4 and len({r['entity'] for r in refs})==len(refs)
   assert all(re.fullmatch('R[0-9]+',r['entity']) and isinstance(r['own_query'],str) and r['own_query'] for r in refs)
   # Count even generic/UNKNOWN heads: the registered full capture explicitly
   # retains their calls and marks availability, not silent budget filtering.
   n=len(refs);total+=n;byplatform[row['platform']]+=n;hist[str(n)]=hist.get(str(n),0)+1
   work.append({'id_PRIVATE':sid,'platform':row['platform'],'graph_sha256':digest,'reference_count':n,
    'references':[{'entity':r['entity'],'own_query_sha256':hashlib.sha256(r['own_query'].encode()).hexdigest(),
     'head_sha256':hashlib.sha256(json.dumps(r.get('head'),sort_keys=True).encode()).hexdigest()} for r in refs]})
  dump(out/'WORKLIST_PRIVATE.json',work)
  result={'state':'R51_FULL5912_PURE_TEXT_REFERENCE_BUDGET_LOCKED_NO_GT','n':5912,
   'manifest_sha256':sha(ROOT/MANIFEST),'pure_text_seal_sha256':sha(ROOT/TEXT/'SEALED.json'),
   'registered_reference_calls':total,'max_grounder_forwards':total,'new_subject_forwards':0,
   'reference_calls_by_platform':byplatform,'reference_count_histogram':hist,
   'UNKNOWN_heads_not_filtered_out':True,'counted_all_graph_references_not_old_detector_worklist':True,
   'model_forward_or_capture_authorized_by_this_result':False,'method_adoption_claimed':False,
   'GT_read':False,'raw_media_or_source_predictions_read':False,'seconds':time.perf_counter()-start}
  dump(out/'RESULT.json',result)
  dump(out/'SEALED.json',{'state':'R51_FULL5912_PURE_TEXT_REFERENCE_BUDGET_COMPLETE_NO_GT','n':5912,
   'plan_sha256':sha(pp),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_cases':len(work),
   'automatic_retry':False,'GT_read':False});raise


if __name__=='__main__':main()
