"""Finite R14 public export builder. No upload/network; NEVER run before readiness.

A root-hashed readiness file must bind BOTH actual postrelease scores, reviewed
report/decision, and final independent scientific review. No arbitrary directory
copy; all code is finite byteexact allowlist and all data is whitelist projection.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,importlib.util,io,json,re,shutil,sys,tokenize,uuid
ROOT=Path(__file__).resolve().parents[1]
PUB=ROOT/'research/round14/publication';sys.path.insert(0,str(PUB))
from export_projection import (STUDY_PREFIX,alias_metric_rows,project_protocol,
                               sanitize_scalar_tree,private_scalar_key,DROP)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def resolve(path):
 p=Path(path);return p if p.is_absolute() else ROOT/p

def artifact_file(item):
 if not {'path','sha256'}<=set(item):raise ValueError('Explicit artifact path/SHA required')
 p=resolve(item['path'])
 if sha(p)!=item['sha256']:raise ValueError('Required final artifact hash mismatch')
 return p

def pointer(value,path):
 parts=path if isinstance(path,list) else (path.lstrip('/').split('/') if path.startswith('/') else path.split('.'))
 for key in parts:
  if isinstance(key,str):key=key.replace('~1','/').replace('~0','~')
  value=value[int(key)] if isinstance(value,list) else value[key]
 return value

def assertions(value,expected):
 if not isinstance(expected,dict) or not expected:raise ValueError('Actual review assertions required; no fabricated PASS')
 for path,want in expected.items():
  if pointer(value,path)!=want:raise ValueError('Final readiness/review assertion failed')

def sealed_scores(item,state,n):
 base=resolve(item['directory']);sealpath=base/'SEALED.json'
 if sha(sealpath)!=item['seal_sha256']:raise ValueError('Score seal binding')
 seal=read(sealpath)
 if seal.get('state')!=state or seal.get('n')!=n or not seal.get('files'):raise ValueError('Final complete postrelease score scope')
 for name,h in seal['files'].items():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts or sha(base/p)!=h:raise ValueError('Score member mismatch')
 return base,seal

def readiness(path,digest):
 if sha(path)!=digest:raise ValueError('Readiness SHA mismatch')
 r=read(path)
 if r.get('schema')!='R14_finite_public_export_readiness_v1' or r.get('build_authorized') is not True:
  raise ValueError('No root build authorization; preparation is not readiness')
 if r.get('builder_sha256')!=sha(Path(__file__)) or r.get('projection_helper_sha256')!=sha(PUB/'export_projection.py'):raise ValueError('Tested builder/helper hash mismatch')
 if r.get('public_prefix')!=STUDY_PREFIX or r.get('both_branches_scores_final') is not True or r.get('root_report_decision_final') is not True:
  raise ValueError('Incomplete publication gates')
 # Verify complete post-GT scores before decoding any task scalar payload.
 qb,qs=sealed_scores(r['quad_scores'],'sealed_R14_Quad_post_both_branches_CPU_score',76)
 db,ds=sealed_scores(r['drone_scores'],'sealed_R14_Drone_postrelease_CPU_scoring',64)
 release=artifact_file(r['both_score_release']);rr=read(release)
 if rr.get('CPU_SCORE_RELEASE') is not True or rr.get('both_branches_complete_and_sealed') is not True or rr.get('stage')!='R14_both_branches_complete_before_target_GT':raise ValueError('Scores lack both-branch GT release')
 if qs['release_sha256']!=sha(release) or ds['release_sha256']!=sha(release):raise ValueError('Scores not bound to actual shared GT release')
 finaldir=resolve(r['drone_final_prediction_directory']);finalsealpath=finaldir/'SEALED.json'
 finalseal=read(finalsealpath)
 if not re.fullmatch(r'crossplatform/round14/drone_association64_attempt[0-9]{3}',r['drone_final_prediction_directory']):raise ValueError('Root must specify actual final Drone attempt')
 if finalseal.get('state')!='sealed_R14_Drone64_seven_direct_arms_before_GT' or finalseal.get('n_cases')!=64 or ds['drone_prediction_seal_sha256']!=sha(finalsealpath):raise ValueError('Final Drone score/prediction attempt mismatch')
 finalplan=artifact_file(r['drone_final_plan'])
 if finalseal['plan_sha256']!=sha(finalplan):raise ValueError('Final Drone plan/prediction binding')
 report=artifact_file(r['root_report']);decision=artifact_file(r['root_decision'])
 if not {'quad_formal','drone_formal','input_exposure','model_context'}<=set(r['plans']):raise ValueError('Core frozen protocol projections required')
 for item in r['plans'].values():artifact_file(item)
 if artifact_file(r['plans']['drone_formal']).resolve()!=finalplan.resolve():raise ValueError('Public final Drone protocol is not actual scored attempt')
 qfinal=resolve(rr['quad_prediction_dir']);qseal=read(qfinal/'SEALED.json')
 if qs['quad_prediction_seal_sha256']!=sha(qfinal/'SEALED.json') or qseal['plan_sha256']!=sha(artifact_file(r['plans']['quad_formal'])):raise ValueError('Quad plan/prediction/score mismatch')
 reviewpath=artifact_file(r['final_scientific_review']);review=read(reviewpath)
 assertions(review,r['final_scientific_review']['required_assertions'])
 if r.get('scientific_review_binds_report_and_decision') is not True:raise ValueError('Final review must bind actual report and decision')
 for publickey,target in [('review_report_sha_field',sha(report)),('review_decision_sha_field',sha(decision))]:
  if pointer(review,r[publickey])!=target:raise ValueError('Scientific review is stale relative to report/decision')
 specpath=artifact_file(r['finite_manifest']);spec=read(specpath)
 if spec['public_prefix']!=STUDY_PREFIX or spec['state']!='PREPARATION_ONLY_not_built_not_uploaded_not_final_manifest':raise ValueError('Finite code spec')
 required=artifact_file(r['required_inputs_spec'])
 if read(required)['public_prefix']!=STUDY_PREFIX:raise ValueError('Required-input spec')
 return r,spec,(qb,qs),(db,ds),report,decision,reviewpath


def public_summary(result,branch):
 common=('state','n','summary','paired_contrasts','geometry_cost_counts','source_checkpoint_sha256','scope')
 extra=('A_nonzero_signal_n','per_scene','mechanism_counts','parameter_kernel_numerical','seconds_scoring') if branch=='quad' else ('cohort_counts','behavioral_diagnostics','seconds')
 return sanitize_scalar_tree({key:result[key] for key in (*common,*extra) if key in result})

def public_tails(result,branch,alias):
 out=[]
 if branch=='quad':
  for tail in result['signed_tails_PRIVATE']:
   row={k:tail[k] for k in ('population','arm','reference','selection')}
   for key,pub in [('largest_negative_PRIVATE','largest_negative'),('largest_positive_PRIVATE','largest_positive')]:
    row[pub]=[{'case_alias':alias[x['pair_id']],'delta_iou':float(x['delta_iou'])} for x in tail[key]]
   out.append(row)
 else:
  allowed=set(alias.values())
  for tail in result['signed_alias_tails']:
   row={k:tail[k] for k in ('cohort','arm','reference')}
   for key in ('negative','positive'):
    if any(x['case_alias'] not in allowed for x in tail[key]):raise ValueError('Non-public Drone tail alias')
    row[key]=[{'case_alias':x['case_alias'],'delta_iou':float(x['delta_iou'])} for x in tail[key]]
   row['postscore_effect_conditioned_not_teacher']=True;out.append(row)
 return sanitize_scalar_tree(out)


def audit_projection(item):
 path=artifact_file(item);original=read(path)
 if item.get('required_assertions'):assertions(original,item['required_assertions'])
 output={'original_sha256':sha(path),'projection':'finite scalar summary; original private artifact remains local'}
 for public,source in item['scalar_field_map'].items():
  if private_scalar_key(public):raise ValueError('Private index/path key requested in public audit')
  value=pointer(original,source)
  if isinstance(value,str):
   if public not in ('state','status') or not re.fullmatch(r'[A-Za-z0-9_.: -]{1,240}',value):raise ValueError('Free-form audit text disallowed')
  elif value is not None and not isinstance(value,(bool,int,float)):raise ValueError('Audit scalar map may not expose rows/arrays/maps')
  output[public]=value
 return sanitize_scalar_tree(output)


def history_scalar(sources, descriptor):
 # Finite typed transforms only, over already SHA-verified evidence. No eval,
 # arbitrary formulas, raw answer access, or synthetic success constants.
 if isinstance(descriptor,(str,list)):return pointer(sources['primary'],descriptor)
 if not isinstance(descriptor,dict) or set(descriptor)-{'source','pointer','op','equals'}:
  raise ValueError('Unsupported history scalar descriptor')
 source=sources[descriptor.get('source','primary')];operation=descriptor.get('op','identity')
 if operation=='count_default_zero':
  parts=descriptor['pointer']
  if not isinstance(parts,list) or not parts:raise ValueError('Count lookup requires explicit key path')
  parent=pointer(source,parts[:-1]) if len(parts)>1 else source
  if not isinstance(parent,dict):raise ValueError('Histogram count parent must be dict')
  value=parent.get(parts[-1],0)
  if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Histogram entry is not a nonnegative count')
  return value
 value=pointer(source,descriptor['pointer'])
 if operation=='identity':return value
 if operation=='not':
  if not isinstance(value,bool):raise ValueError('Boolean inversion needs actual bool')
  return not value
 if operation=='equals':
  if not isinstance(value,(bool,int,float)) or not isinstance(descriptor['equals'],(bool,int,float)):
   raise ValueError('Equality proof must be scalar numeric/boolean')
  return value==descriptor['equals']
 raise ValueError('Unsupported history scalar transform')


def history_projection(events):
 required={'drone_formal001_interrupted','drone_format_fix_review','drone_engineering001','drone_engineering002','drone_formal002_budget_bottleneck','drone_attempt003_preGT_authorization','drone_formal001_closed_inventory','drone_engineering003','drone_attempt004_preGT_authorization','drone_engineering004_metadata_failure','drone_engineering004b'}
 if not required<={e['event_alias'] for e in events}:raise ValueError('Required failed/interrupted/preflight lineage missing')
 out=[]
 allowed={'completed_cases','expected_cases','new_model_calls','reused_calls','failed_cases','logical_calls','no_GT_at_event','passed','checked_count','visual_calls','visual_cap','visual_no_eos_n','valid_visual_calls','combined_source_fallback_cases','reference_contract_failed_calls','progress_reused_calls','actual_closed_calls','visual_eos_n','formal_started','reported_seal_cases','actual_case_files','visual_resolved_calls','main_mask_consensus_cases','combined_changed_cases','renamed_NPZ_fields_compared','raw_answer_pairs_compared','outputs_identical','algorithm_changed'}
 reasons={'strict_JSON_codefence_interface','root_interrupted_before_GT','format_fix_only','engineering_preflight','engineering_interface_failure','engineering002_pass','visual_generation_budget_reference_contract','preGT_budget_reference_interface_fix','engineering003_pass','nonstring_reference_slot_discards_valid_target','preGT_reference_slot_type_interface_fix','engineering_count_variable_shadow','engineering004b_metadata_only_rename_pass'}
 for event in events:
  path=artifact_file(event);original=read(path)
  if event.get('required_assertions'):assertions(original,event['required_assertions'])
  if event['reason_code'] not in reasons:raise ValueError('History reason must be fixed format/scope code, never raw answer/reason text')
  sources={'primary':original};additional_sha={}
  for name,item in event.get('additional_sources',{}).items():
   if name=='primary' or not re.fullmatch(r'[A-Za-z0-9_]+',name):raise ValueError('History source alias')
   extra=artifact_file(item);sources[name]=read(extra);additional_sha[name]=sha(extra)
   if item.get('required_assertions'):assertions(sources[name],item['required_assertions'])
  row={'event_alias':event['event_alias'],'reason_code':event['reason_code'],'original_sha256':sha(path)}
  if additional_sha:row['additional_source_sha256']=additional_sha
  for key,pointerpath in event['numeric_field_map'].items():
   if key not in allowed:raise ValueError('History counts-only whitelist')
   value=history_scalar(sources,pointerpath)
   if value is not None and not isinstance(value,(int,float,bool)):raise ValueError('Raw history record/answer forbidden')
   row[key]=value
  out.append(row)
 by_event={x['event_alias']:x for x in out}
 expected001={'completed_cases':5,'expected_cases':64,'progress_reused_calls':20}
 expected002={'visual_calls':128,'visual_cap':64,'visual_no_eos_n':61,'valid_visual_calls':0,'combined_source_fallback_cases':64}
 for key,value in expected001.items():
  if by_event['drone_formal001_interrupted'].get(key)!=value:raise ValueError('001 history counts do not match frozen interruption evidence')
 for key,value in expected002.items():
  if by_event['drone_formal002_budget_bottleneck'].get(key)!=value:raise ValueError('002 history counts do not match frozen no-GT budget audit')
 expected_closed={'actual_closed_calls':39,'new_model_calls':15,'reused_calls':24}
 for key,value in expected_closed.items():
  if by_event['drone_formal001_closed_inventory'].get(key)!=value:raise ValueError('001 actual closed inventory must not be replaced by earlier progress20')
 expected003={'completed_cases':2,'new_model_calls':10,'reused_calls':4,'visual_calls':4,'visual_eos_n':4,'formal_started':False}
 for key,value in expected003.items():
  if by_event['drone_engineering003'].get(key)!=value:raise ValueError('003 was engineering-only, not a completed formal run')
 expected004={'completed_cases':2,'new_model_calls':10,'reused_calls':4,'reported_seal_cases':4,'actual_case_files':2,'formal_started':False}
 for key,value in expected004.items():
  if by_event['drone_engineering004_metadata_failure'].get(key)!=value:raise ValueError('004 engineering metadata failure must not be called4cases/PASS')
 expected004b={'completed_cases':2,'new_model_calls':10,'reused_calls':4,'checked_count':1142,'outputs_identical':True,'algorithm_changed':False}
 for key,value in expected004b.items():
  if by_event['drone_engineering004b'].get(key)!=value:raise ValueError('004b actual audited metadata-only repair evidence required')
 return out


def report_projection(text,report_path,mapping):
 # Only link destinations change; scientific prose is not summarized/repaired.
 def link(match):
  image,label,target=match.groups();clean=target.split('#',1)[0]
  if clean.startswith(('https://','http://')):return match.group(0)
  candidate=Path(clean);candidate=candidate if candidate.is_absolute() else report_path.parent/candidate
  candidate=candidate.resolve()
  if candidate in mapping and not image:return '['+label+']('+mapping[candidate]+')'
  return label+' (private evidence retained locally; not exported)'
 return re.sub(r'(!?)\[([^\]]*)\]\(([^)]+)\)',link,text)


def private_markers(registries):
 ids=set();text=set()
 def visit(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,str):
     n=re.sub('[^a-z0-9]','',k.casefold())
     if n in ('id','aid','bid','sampleid') and len(v)>=8:ids.add(v)
     if n in ('scene','sceneprivate','caption','utterance') and len(v)>=8:text.add(v)
    if isinstance(v,(dict,list)):visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 if not registries:raise ValueError('Input-only privacy marker registries required')
 required={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
 for item in registries:
  rows=read(artifact_file(item))
  if not isinstance(rows,list) or not rows or any(not isinstance(row,dict) or set(row)!=required for row in rows):raise ValueError('Privacy markers must come only from source9-key input registries, never GT/index/rawcalls')
  visit(rows)
 return ids,text


def code_contains_private_host_path(content):
 # These two exact string literals implement generic privacy guards / a
 # synthetic fixture. They are not user host paths. Only .py gets this typed
 # exception; reports, JSON and tables retain the strict original scan.
 generic={'/home/','/home/private/model'}
 fragments=('/home/',('/'+'Users'+'/'),('file'+':'+'//'))
 tokens=[]
 for token in tokenize.generate_tokens(io.StringIO(content).readline):
  if token.type==tokenize.STRING:
   try:value=ast.literal_eval(token.string)
   except (ValueError,SyntaxError):value=None
   if isinstance(value,str) and any(p in value for p in fragments):
    if value not in generic:return True
    token=token._replace(string=repr('GENERIC_PRIVACY_SENTINEL_LITERAL'))
  tokens.append(token)
 # Comments and f-strings remain scanned, as do non-whitelisted literals.
 remaining=tokenize.untokenize(tokens)
 return any(p in remaining for p in fragments)


def verify_public_tree(base,ids,private_text):
 for p in base.rglob('*'):
  if not p.is_file():continue
  if p.suffix.lower() in ('.npz','.npy','.pth','.pt','.safetensors','.jpg','.png','.bin','.zip'):raise ValueError('Forbidden raw/media/model artifact')
  if p.name.endswith('.csv.gz'):content=gzip.decompress(p.read_bytes()).decode()
  else:
   try:content=p.read_text()
   except UnicodeDecodeError:raise ValueError('Unexpected public binary')
  if any(re.search(r'(?<![A-Za-z0-9])'+re.escape(v)+r'(?![A-Za-z0-9])',content) for v in ids):raise ValueError('Raw input identifier leak in finite export')
  low=' '.join(content.casefold().split())
  if any(' '.join(v.casefold().split()) in low for v in private_text):raise ValueError('Physical scene or literal caption leak')
  if (code_contains_private_host_path(content) if p.suffix=='.py' else any(v in content for v in ('/home/',('/'+'Users'+'/'),('file'+':'+'//')))):raise ValueError('Private host path leak')
  if p.suffix=='.json':
   def inspect(x):
    if isinstance(x,dict):
     for key,value in x.items():
      if private_scalar_key(key):raise ValueError('Private scalar/index field survived projection')
      inspect(value)
    elif isinstance(x,list):
     for value in x:inspect(value)
   inspect(json.loads(content))


def build(a):
 r,spec,quad,drone,report,decision,review=readiness(a.readiness,a.readiness_sha256)
 output=a.output.resolve()
 if not output.is_relative_to(ROOT/'research/round14') or output.exists():raise ValueError('New R14 output directory only; never overwrite old studies')
 aliaspath=resolve(r['private_alias_map_output'])
 if aliaspath.is_relative_to(output) or aliaspath.exists():raise ValueError('Private alias map must be new and OUTSIDE public package')
 stage=output.with_name(output.name+'.building_'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
 original_sha={};mapping={}
 try:
  # Exact finite code list only. No globs, recursive source tree copy or asset reads.
  for entry in spec['code_byteexact_allowlist']:
   source=resolve(entry['source_workspace_relative']);destination=Path(entry['destination'])
   if destination.is_absolute() or '..' in destination.parts or not str(destination).startswith('code/') or source.suffix not in ('.py','.patch'):raise ValueError('Code allowlist escape')
   if sha(source)!=entry['sha256_at_preparation']:raise ValueError('Source changed since finite spec; root must rebind actual frozen provenance')
   dest=stage/destination;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
   if sha(dest)!=sha(source):raise ValueError('Code byteexact copy failed')
   mapping[source.resolve()]=destination.as_posix();original_sha[destination.as_posix()]=sha(source)
  for entry in r.get('extra_code_history',[]):
   source=artifact_file(entry);dest=Path(entry['destination'])
   if source.suffix not in ('.py','.patch') or dest.is_absolute() or '..' in dest.parts or not str(dest).startswith('history/code/'):raise ValueError('History code finite whitelist')
   target=stage/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);original_sha[dest.as_posix()]=sha(source)
  patch=resolve(spec['upstream_requirement']['compatibility_patch_workspace_relative'])
  if sha(patch)!=spec['upstream_requirement']['patch_sha256']:raise ValueError('Upstream compat patch changed')
  shutil.copyfile(patch,stage/'UPSTREAM_COMPATIBILITY.patch')
  write(stage/'UPSTREAM_PROVENANCE.json',{k:v for k,v in spec['upstream_requirement'].items() if k!='compatibility_patch_workspace_relative'})
  # Task scalars are decoded ONLY after complete readiness returned above.
  qr=read(quad[0]/'RESULTS_PRIVATE.json');dr=read(drone[0]/'RESULTS_PRIVATE.json')
  if qr['n']!=76 or dr['n']!=64:raise ValueError('All cases required')
  qrows=qr['rows_PRIVATE'];drows=dr['rows_PRIVATE']
  qpublic=alias_metric_rows(qrows,platform='quad',expected_n=76,prefix='quad_pair',preserve_ordinal_alias=True)
  qalias={row['pair_id']:pub['case_alias'] for row,pub in zip(qrows,qpublic)}
  dspec=importlib.util.spec_from_file_location('R14_bounded_Drone_scalar_projection',stage/'code/research/round14/drone/scalar_projection.py')
  dm=importlib.util.module_from_spec(dspec);dspec.loader.exec_module(dm);doutputs,dschema=dm.project(dr)
  dalias={row['id']:row['case_alias'] for row in drows}
  write(stage/'scalars/QUAD_CASE_METRICS.json',qpublic)
  for name,data in doutputs.items():(stage/'scalars').mkdir(exist_ok=True);(stage/'scalars'/name).write_bytes(data)
  write(stage/'scalars/DRONE_SCALAR_SCHEMA.json',sanitize_scalar_tree(dschema))
  for extra in r.get('additional_drone_scores',[]):
   eb,es=sealed_scores(extra,'sealed_R14_Drone_postrelease_CPU_scoring',64)
   extra_release=artifact_file(extra['score_release']);erel=read(extra_release)
   if not erel.get('CPU_SCORE_RELEASE') or not erel.get('both_branches_complete_and_sealed') or es['release_sha256']!=sha(extra_release):raise ValueError('Legacy002 score lacks actual post-both-branches release')
   if not re.fullmatch(r'drone_attempt[0-9]{3}',extra['attempt_alias']):raise ValueError('Finite legacy attempt alias')
   legacy=read(eb/'RESULTS_PRIVATE.json');outputs,schema=dm.project(legacy)
   dest=stage/'history/scalars'/extra['attempt_alias'];dest.mkdir(parents=True)
   for filename,blob in outputs.items():(dest/filename).write_bytes(blob)
   write(dest/'SUMMARY.json',public_summary(legacy,'drone'))
   write(dest/'SCOPE.json',{'same64_related_observations_not_new_independent_cases':True,'reason_code':extra['reason_code'],'original_score_seal_sha256':sha(eb/'SEALED.json'),'not_automatically_semantic_method_failure':True})
  for name,result,alias in [('QUAD',qr,qalias),('DRONE',dr,dalias)]:
   write(stage/'scalars'/(name+'_SUMMARY.json'),public_summary(result,name.casefold()))
   write(stage/'scalars'/(name+'_SIGNED_TAILS.json'),public_tails(result,name.casefold(),alias))
  private_alias={'quad':[{'public_alias':qalias[x['pair_id']],'original_pair_id_PRIVATE':x['pair_id'],'A_id_PRIVATE':x['A_id_PRIVATE'],'B_id_PRIVATE':x['B_id_PRIVATE']} for x in qrows],
    'drone':[{'public_alias':x['case_alias'],'original_id_PRIVATE':x['id']} for x in drows]}
  for role,item in r['plans'].items():
   if not re.fullmatch(r'[A-Za-z0-9_-]+',role):raise ValueError('Protocol role')
   path=artifact_file(item);original_plan=read(path)
   safe_resource_names={'config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','chat_template.json','model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors','pytorch_model.bin','special_tokens_map.json','added_tokens.json','tokenizer.model','vocab.txt'}
   resource_hashes={Path(k).name:v for k,v in original_plan.get('model_files_sha256',{}).items() if Path(k).name in safe_resource_names}
   write(stage/'protocol'/(role+'.json'),{'original_sha256':sha(path),'settings':project_protocol(original_plan),'generic_resource_file_sha256':resource_hashes})
  for role,item in r['audits'].items():
   if not re.fullmatch(r'[A-Za-z0-9_-]+',role):raise ValueError('Audit role')
   path=artifact_file(item);write(stage/'audits'/(role+'.json'),audit_projection(item));mapping[path.resolve()]='audits/'+role+'.json'
  write(stage/'HISTORY_COUNTS.json',history_projection(r['history_records']))
  dec=read(decision)
  if not r.get('decision_public_keys'):raise ValueError('Explicit reviewed decision key whitelist required')
  write(stage/'DECISION.json',sanitize_scalar_tree({k:dec[k] for k in r['decision_public_keys']}));mapping[decision.resolve()]='DECISION.json'
  (stage/'REPORT.md').write_text(report_projection(report.read_text(),report,mapping))
  (stage/'README.md').write_text('''# R14：Quad 跨输入回归迁移与 Drone 主体/映射诊断\n\n本包保存已执行协议、完整匿名标量、负例、修订历史和审计投影。结论以 [REPORT](REPORT.md) 和 [DECISION](DECISION.json) 为准，本文不自动宣布方法有效。\n\nQuad 使用冻结 Waymo-only source100。A 为 R13 Quad76 的全部既有不等式 G，B 是同场景一一配对的不同图像及点内容76。B 只用于原源特征/固定自身 q0 读出，不构造 G_B、不参与 A loss/LR/gate。真实末回归层 half/full 更新与精确 frozen-feature kernel 对照并列，另有 additive 和 source-size-normalized 直接迁移。即使 B 未参与拟合，末仿射头更新仍与跨输入 kernel 理论等价；数值差不作为独有学习贡献。\n\nDrone 将旧32开发例与近期内容不同32分开报告；64并集只作描述。冻结角色解析与原3D候选映射均不构造新框、不做参数更新。完整 JSON 的单层 codefence 格式修复发生在 GT 前；工程001、formal001中断、工程002及formal002预算/引用契约瓶颈、003仅工程预检与004的GT前授权/审查等历史以 [计数记录](HISTORY_COUNTS.json) 留存，不公开 raw calls。002的visualcap64下61/128未EOS、0valid visual与combined全source是无GT工程覆盖瓶颈，不能据此宣称语义方法失败；若另有002任务标量，其64例作为相关重复观测另表，不计入最终64分母。001进度记录的20reuse不等于实际闭合调用总量；闭合inventory为39=15new+24reuse，二者分别留存。003仅2例工程预检（10new+4reuse），4/4EOS后仍出现reference槽类型把合法target一起丢弃的接口问题，未启动formal003；004只局部UNKNOWN处理reference内部非string槽，主target严格契约和模型/提示/几何保持锁定。首次004工程的主体分支虽通过，却因runner局部变量shadow把实际2case误记为seal n_cases4，故未启动formal004；10new+4reuse与原错误seal保留为元数据失败，变量名窄修不改变算法，004b工程及后续formal另计。最终attempt由root readiness与实际score/prediction/plan哈希绑定，本文不偷换任何未完成attempt为已通过结果。\n\nAll5912 targetVAL 记录有 R7/R8 历史模型/评分暴露。本轮独立封存不消除旧标签暴露；distinct content 不等于已证明同一物体、不同实例或独立场景。三场景依赖、零信号、NA、全部正负变化均保留，alias 标量不是独立样本数量的扩增。\n\n仅代码/协议/标量可审计导出：未随包发布原图、点云、mask、features、原始GT/预测坐标、raw IDs/captions/calls、权重及私有输入注册。精确预测重放仍需合法取得官方数据、固定资产以及原 source100 checkpoint/私有注册；本包不独立包含全部运行资源。可执行项目代码依赖已纳入 code/，官方3EED checkout与兼容补丁见 UPSTREAM_PROVENANCE.json。\n\n公开标量包含 Quad 全76与 Drone 全64（旧32/近期32清楚分开），没有删除失败、零变化或负例。后续 build 后仍需独立 privacy 审查，再由 root 单独发布；本脚本没有上传功能。\n''')
  write(stage/'PROVENANCE.json',{'public_prefix':STUDY_PREFIX,'readiness_original_sha256':sha(a.readiness),
    'root_report_original_sha256':sha(report),'root_decision_original_sha256':sha(decision),
    'final_scientific_review_original_sha256':sha(review),'quad_score_seal_sha256':sha(quad[0]/'SEALED.json'),
    'drone_score_seal_sha256':sha(drone[0]/'SEALED.json'),'source_code_original_sha256':original_sha,
    'private_input_registries_and_alias_join_not_exported':True,'all76_64_cases_retained':True})
  ids,private_text=private_markers(r['private_marker_registries'])
  verify_public_tree(stage,ids,private_text)
  files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
  if not files:raise ValueError('Empty public package')
  write(stage/'MANIFEST.json',{'state':'built_R14_finite_public_candidate_awaiting_independent_privacy_review_not_uploaded',
    'public_prefix':STUDY_PREFIX,'n_files_excluding_manifest':len(files),'files':files,
    'quad_alias_cases':76,'drone_alias_cases':64,'no_upload_executed':True})
  write(aliaspath,private_alias);stage.rename(output)
  print(json.dumps({'state':'built_candidate_not_uploaded','output':str(output),'n_files':len(files)+1,'manifest_sha256':sha(output/'MANIFEST.json')}))
 except Exception:
  # Preserve failed staging for local review; never upload or silently relabel PASS.
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--readiness',type=Path,required=True);p.add_argument('--readiness-sha256',required=True);p.add_argument('--output',type=Path,required=True);build(p.parse_args())
