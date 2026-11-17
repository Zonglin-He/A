from common134 import *
import csv,io
O134=H134/'publication/public_export_attempt001'
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H134/'SCALAR_AUDIT.json')['n']==1798;O134.mkdir(parents=True,exist_ok=False);Q=H134/'scores_attempt001';rows=[r for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(p)];frames=sorted({r['frame_PRIVATE'] for r in rows});contents=sorted({r['content_group_PRIVATE'] for r in rows});scenes={t:sorted({r['scene_PRIVATE'] for r in rows if r['target']==t}) for t in ['waymo','drone','quad']};mapping={}
 for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  pub=[]
  for i,r in enumerate(rd(p)):
   ident={'row':f'ROW{i+1:04d}','frame_cluster':f'FRAME{frames.index(r["frame_PRIVATE"])+1:04d}','content_cluster':f'CONTENT{contents.index(r["content_group_PRIVATE"])+1:04d}','scene_cluster':f'CLUSTER{scenes[r["target"]].index(r["scene_PRIVATE"])+1:03d}'};mapping[(r['source'],r['target'],r['id_PRIVATE'])]=ident;d={k:v for k,v in r.items() if not k.endswith('_PRIVATE') and k not in ['IoUs','diagnostics']};d.update(ident);d.update({k+'__IoU':v for k,v in r['IoUs'].items()})
   for arm,diag in r['diagnostics'].items():d.update({arm+'__'+k:v for k,v in diag.items()})
   pub.append(d)
  table(O134/'tasks'/p.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),pub)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O134/'results'/(name+'.csv'),rd(Q/(name+'.json')))
 pub=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):
  d=dict(r);d['scene_cluster']=f'CLUSTER{scenes[r["target"]].index(d.pop("scene_PRIVATE"))+1:03d}';pub.append(d)
 table(O134/'results/ALL_SCENE_ROWS.csv',pub)
 for name in ['CHANGED_CASES','R132_ALL76_CASES','OLD_NO_QUERY_CHANGE508','PAIR_SIGNAL_ROWS','PROCESS_ROWS']:
  pub=[]
  for r in rd(H134/(name+'_PRIVATE.json')):
   d={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};d.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);pub.append(d)
  table(O134/'diagnostics'/(name+'.csv'),pub)
 table(O134/'diagnostics/COST_SUMMARY.csv',rd(H134/'COST_SUMMARY.json'))
 for name in ['PREFLIGHT','PREPARATION_SUMMARY','ENGINEERING','NATIVE_AUDIT','SCALAR_AUDIT','CASE_SUMMARY','R132_CASE_SUMMARY','NO_CHANGE_SUMMARY','SIGNAL_SUMMARY','DECISION']:dump(O134/(name+'.json'),rd(H134/(name+'.json')))
 for p in (H134/'experiment').glob('*.json'):dump(O134/'config'/p.name,rd(p))
 for name in ['REPORT.md','FINDINGS.md','LITERATURE.md']:(O134/name).write_text((H134/name).read_text())
 for p in list(H134.glob('*.py'))+[R134/'tools/experiment_guard_round134.py']:
  out=O134/'own_code'/p.relative_to(R134);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(p.read_text().replace(str(R134),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for p in sorted((H134/'job_receipts').glob('*.json')):
  r=rd(p);receipts.append({k:r[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(r['command'][1]).name,'engineering':'--engineering' in r['command']})
 dump(O134/'RESOURCE_RECEIPTS.json',receipts)
 (O134/'REPRODUCTION.md').write_text('''# R134 reproduction and boundaries

Depends on immutable R13354343a1914d8ff3af1d51cf705979876c2738563 and R132972f5055f5213a8a362ea85c4d7ed9e8f5593d1f, with R131/R130/R129/R127/R123/R97/R91 dependencies documented there. Upstream https://github.com/worldbench/3EED . Licensed paired data, private features/coordinates/IDs/weights/states are not exported.

Order: GT-free preflight134 -> prepare134 frozen ordinal evidence/direct projection -> model lock -> run134 --engineering -> formal lock -> run134 -> seal all endpoints -> audit134 -> score134 -> audit_scalars134 -> analyze134 -> report134 -> export134 -> audit_public134. Model/CPU jobs run via experiment_guard_round134.py, one exclusive GPU, old scopes closed. Original A is never inferred again; original B features are reused or deterministically reconstructed as documented. All1798 rows remain;381 new fit inputs with7157 frozen pairs, including114 previously C=S.11 engineering examples span all6directions and both B feature paths.

The actual deployed full FAB readout is used in the objective and final prediction. Only pointUp4896 changes; same original three steps, per-step source radius, feature keep1,9-level Armijo. Primary partial_pair_F adds mean hinge on frozen AB-agreed, initial-FAB-reversed pairs within existing C. Margins min(gapA,gapB,existingkappa); no target threshold selection. No unique target label or all-other-negative assumption. Pair-only and permuted-pair updates, zero/direct/oldpartial3, exact pair-constrained no-update projection and oldpartial3+projection are all retained. Wrong graph retains C/S/margin multiset/group-size bins but not initial hinge or semantic mass. No claim these are official external baselines.

All predictions are frozen before cached R131 fixed-box IoUs. No new raw GT geometry, new source training, external expert, B3, source calibration, view or FINAL. Target scores/case GT only evaluate and describe, never construct/update/select a deployment output. Existing exposure and pooled source-own three-source calibration remain; no fresh blind validation or per-direction source-only label guarantee. Candidate geometric groups and pairwise box-IoU preference do not certify object identity.

All thirteen-arm task scalars, all six directions, targets, anonymous shared content/frame/recording clusters, old76cases and old508unchanged updates are exported. A single Drone recording does not permit population scene CIs. Exact states/step arithmetic, readout and objective replay plus convex projection KKT/scalars/privacy are independently checked. Full decoder backward is not rerun on CPU for all inputs; engineering directional derivatives check the new path.

Only own code/config/anonymous scalar results are public. All immutable remote bytes/tree/parent verified before branch CAS. No raw media/descriptions/coordinates/query IDs/features/weights/secrets/third-party code. A task-relevance ablation was fixed before any task scoring: use new pair update only if a frozen AB order opposes initial final top; otherwise retain oldpartial3, with matched gated direct/wrong readouts. No hidden follow-up sweep; retain old working version if this bounded repair fails.
''')
 dump(O134/'MANIFEST.json',{'prefix':'studies/3eed_retained_candidate_ordinal_TTA_repair_20261117','files':[{'path':str(p.relative_to(O134)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O134.rglob('*')) if p.is_file()]});print('EXPORT134',len([p for p in O134.rglob('*') if p.is_file()]))
if __name__=='__main__':main()
