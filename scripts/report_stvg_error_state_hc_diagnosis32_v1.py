"""Anonymous cached-score report; no new labels, prediction fit or GT loading."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/stvg_error_state_hc_diagnosis32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/hc_diagnosis32'
GROUPS=['WI','CP','MX','CI','UN']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ep(r,base,name,key):
 v=r['values'][base][name];return v[key]if key in ['v','s']else v['supports'][key]['s']
def pair(v):
 v=np.array([a for a in v if a is not None],float)
 if not len(v):return dict(mean=None,CI95=None,parents=0)
 ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
 loo=(v.sum()-v)/(len(v)-1)if len(v)>1 else np.array([np.nan])
 return dict(mean=float(v.mean()),CI95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),parents=len(v),positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),neutral=int((abs(v)<=1e-10).sum()),severe_loss_below_minus5pp=int((v< -5).sum()),leave_one_saved_mean_range=[float(loo.min()),float(loo.max())]if len(v)>1 else None)
def contrast(rows,base,choose,control,key):
 vals=[]
 for r in rows:
  a,b=ep(r,base,choose(r),key),ep(r,base,control,key);vals.append(None if a is None or b is None else 100*(a-b))
 return pair(vals)
def fmt(p):return'UNKNOWN'if p['mean']is None else f"{p['mean']:+.6f} [{p['CI95'][0]:+.6f}, {p['CI95'][1]:+.6f}]"

def main():
 x=read(P/'RESULTS.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
 rows=x['rows'];old_config=read(ROOT/'results/stvg_error_state/2026-10-18/cached32/ACTUAL_PROTOCOL.json')
 old_pred={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/cached32/RESULTS.json')['rows']}
 diagnostics={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];diagnostics[ds]={}
  for base in ['Before','Source']:
   def groupmask(r,wi=True):
    present=[g for g in GROUPS if r['common_reference_counts'][g]>0];allmask=(1<<len(present))-1;w=(1<<present.index('WI'))if'WI'in present else 0
    return'groups_'+str(w if wi else allmask^w)
   hard={name:{control:{key:contrast(rr,base,lambda r,wi=wi:groupmask(r,wi),control,key)for key in ['v','unobserved']}for control in ['Base','geometry_0']}for name,wi in [('WI_only',True),('nonWI_only',False)]}
   geometry={str(i):{key:contrast(rr,base,lambda r,i=i:'geometry_'+str(i),'Base',key)for key in ['v','unobserved']}for i in range(4)}
   conditional={g:{key:pair([r['category_repair_shapley_pp'][base][key][g]for r in rr if r['common_reference_counts'][g]>0 and r['category_repair_shapley_pp'][base][key]is not None])for key in ['v','unobserved']}for g in GROUPS}
   diagnostics[ds][base]=dict(hard_group_repairs=hard,geometry_hybrids_vs_Base=geometry,category_present_parent_conditional=conditional,conditional_not_additive_to_cohort=True)
 examples={}
 for alias in ['H003','H005','H010']:
  r=old_pred[alias];examples[alias]={ref:{key:r['values']['Before'][ref]['GlobalError']['clocks']['final_MBR']['v']-r['values']['Before'][ref]['Base']['clocks']['final_MBR']['v']if key=='v_fraction'else 100*(r['values']['Before'][ref]['GlobalError']['supports']['unobserved_decoder']['s']-r['values']['Before'][ref]['Base']['supports']['unobserved_decoder']['s'])for key in ['v_fraction','unobserved_pp']}for ref in ['DINO_common','GT_common']}
 put(P/'DIAGNOSTICS.json',dict(status='complete',groups=diagnostics,illustrative_saved_cases=examples,no_new_GT_or_scoring=True,no_selection_for_deployment=True,leave_one_is_saved_scores_not_stream_replay=True))
 completion=read(D/'CPU_COMPLETION.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
 actual=dict(status='complete',count=32,focus='HC16; Vid16 context',primary='original Before/final MBR; Source sensitivity',historically_exposed=True,backbone=old_config['backbone'],source_domains_and_checkpoint_hashes=old_config['source_domains_and_checkpoint_hashes'],cached_DINO_expert=old_config['cached_DINO_expert'],original_capture_reused_no_new_forward=True,
  common_references=dict(hc2=43,vidstg=24),old_labels='model-assisted WI/CP/MX/CI/UN only; apparent identity not GT geometry or independent human truth',unknown_semantic_role_queries=6,all_role_unknown_slots_overridden_UN_original_codes_retained=True,
  binding_semantics=read(D/'BINDING_SEMANTICS_CLARIFICATION.json'),
  category_game='all subsets of categories present; DINO_common -> same-position GT_common repair; exact finite Shapley',
  geometry_game='2x2 DINO/GT centers and sizes, equal-weight constant LS; exact two-factor Shapley',
  scopes='same estimated delta under global, four-planned-position hull, original admitted decoder-node masks; none uses GT for mask geometry',
  supports='known actual decoder frames exclude all four planned slots; within/outside hull weighted contributions use the original per-query total unobserved denominator',
  overlapping_estimands_not_additive_causes=True,reference_geometry_means_anchor_weighted=True,repair_effects_parent_weighted=True,
  bootstrap=dict(draws=10000,seed=20261018,unit='parent, one query each',confidence=.95,multiplicity_adjusted=False,fixed_saved_inherited_outputs_not_replay=True),
  oracle_geometry_and_common_availability_not_deployable=True,all32_and_noops_unknown_retained=True,
  new_pixels_visually_inspected=0,new_identity_labels=0,new_GPU_models_experts_training_Student_updates=0,
  CPU_cost=dict(candidate_generation_seconds=bar['CPU_wall_seconds'],dense_score_seconds=completion['CPU_score_seconds'],not_online_end_to_end_cost=True),
  pins=read(D/'RUNTIME_LOCK.json')['pins'],plan_sha256=sha(D/'PLAN_LOCK.json'),prediction_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),production_registries_unchanged=True,
  conclusion='category-conditioned oracle reference repair, geometric-coordinate and application-scope diagnostics only; no filter, chosen hybrid, winner or OPD dispatch')
 put(P/'ACTUAL_PROTOCOL.json',actual);put(P/'VERIFICATION.json',dict(status='pass',count=32,independent_audit=audit,root_metric_endpoint_identity_Shapley_checks=completion['metric_endpoint_identity_and_Shapley_checks'],old_model_assisted_labels_not_new_human=True,all_candidates_sealed_before_new_dense_score=True,GT_coordinates_media_q_weights_raw_tensors_excluded=True))
 fig,axs=plt.subplots(2,3,figsize=(17,11));fig.subplots_adjust(left=.065,right=.985,top=.88,bottom=.12,wspace=.35,hspace=.46)
 hc=x['groups']['hc2']['Before'];colors=['#246e9b','#b36333']
 aa=axs[0,0]
 for j,g in enumerate(GROUPS):
  for k,key in enumerate(['v','unobserved']):
   p=hc['category_oracle_repair_shapley'][key][g]['cohort_contribution'];y=4-j+(k-.5)*.22
   aa.errorbar(p['mean'],y,xerr=[[p['mean']-p['CI95'][0]],[p['CI95'][1]-p['mean']]],fmt='o',color=colors[k],capsize=2,label=['Full Δv','Unobserved Δs'][k]if j==0 else None)
 aa.axvline(0,color='.5',lw=.8);aa.set_yticks(range(5),list(reversed(GROUPS)));aa.legend(fontsize=8);aa.set_title('A  HC category oracle-repair Shapley (pp)',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=axs[0,1]
 for j,key in enumerate(['v','unobserved']):
  vals=[diagnostics['hc2']['Before']['geometry_hybrids_vs_Base'][str(i)][key]['mean']for i in range(4)]
  aa.bar(np.arange(4)+(j-.5)*.32,vals,width=.3,color=colors[j],label=['Full Δv','Unobserved Δs'][j])
 aa.axhline(0,color='.5',lw=.8);aa.set_xticks(range(4),['DINO','GT center','GT size','GT both'],rotation=15,fontsize=8);aa.legend(fontsize=8);aa.set_title('B  HC geometry hybrids vs Base (pp)',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=axs[0,2];counts=hc['category_oracle_repair_shapley']['v'];quality=hc['reference_geometry'];vals=[quality[g]['reference_IoU']for g in GROUPS]
 bars=aa.bar(range(5),vals,color='#5c8b77');aa.set_ylim(0,1);aa.set_xticks(range(5),[f"{g}\n{counts[g]['reference_count']}a / {counts[g]['present_parents']}p"for g in GROUPS],fontsize=8)
 for b in bars:aa.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f'{b.get_height():.3f}',ha='center',fontsize=8)
 aa.set_title('C  HC common-reference IoU; anchors / parents',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=axs[1,0]
 for j,scope in enumerate(['global','planned_hull','admitted_nodes']):
  vals=[hc['scope'][ref][scope]['unobserved']['mean']for ref in ['DINO_full','DINO_common','GT_common']]
  aa.bar(np.arange(3)+(j-1)*.25,vals,width=.23,color=['#246e9b','#b88739','#6e7e87'][j],label=['Global','Planned hull','Admitted nodes'][j])
 aa.axhline(0,color='.5',lw=.8);aa.set_xticks(range(3),['DINO full','DINO common','GT common'],rotation=15,fontsize=8);aa.legend(fontsize=7.5);aa.set_title('D  HC same-delta scope: unobserved Δs (pp)',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=axs[1,1]
 inside=[];outside=[]
 for ds in ['hc2','vidstg']:
  for ref in ['DINO_full','DINO_common','GT_common']:
   parts=x['groups'][ds]['Before']['scope'][ref]['global']['signed_unobserved_parts'];inside.append(parts['inside_hull']['mean']);outside.append(parts['outside_hull']['mean'])
 aa.bar(range(6),inside,color='#437e9d',label='Inside contribution')
 bottom=np.where(np.sign(inside)==np.sign(outside),inside,0);aa.bar(range(6),outside,bottom=bottom,color='#be8b45',label='Outside contribution')
 aa.axhline(0,color='.5',lw=.8);aa.set_xticks(range(6),['H full','H common','H GT','V full','V common','V GT'],rotation=30,fontsize=7.5);aa.legend(fontsize=7.5);aa.set_title('E  Additive original-denominator Δs parts (pp)',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 aa=axs[1,2]
 for k,base in enumerate(['Before','Source']):
  vals=[x['groups']['hc2'][base]['category_oracle_repair_shapley']['unobserved'][g]['cohort_contribution']['mean']for g in GROUPS]
  aa.plot(range(5),vals,'o-'if k==0 else's--',label=base,color=colors[k])
 aa.axhline(0,color='.5',lw=.8);aa.set_xticks(range(5),GROUPS);aa.legend(fontsize=8);aa.set_title('F  HC category unobserved Shapley sensitivity',loc='left',fontsize=10);aa.spines[['top','right']].set_visible(False)
 fig.suptitle('Why HC GT-reference and DINO correction differ: three overlapping controlled contrasts',fontsize=15,y=.97)
 fig.text(.065,.918,'Cached32, HC16 focus; Before/final MBR is primary. Old model-assisted labels only, six unknown-role queries conservatively UN.',fontsize=10)
 fig.text(.065,.045,'All category contributions use the full16-parent denominator. Geometry factors are not identity/extent labels.\nScope contributions add only with each query’s original unobserved denominator; no chosen filter, mask, hybrid, new label or OPD.',fontsize=10)
 fig.savefig(P/'HC_DIAGNOSIS.svg');fig.savefig(D/'HC_DIAGNOSIS.png',dpi=145);plt.close(fig)
 lines=['# Why HC correct-reference correction and actual DINO correction differ','',
 'The cached evidence separates reference mistakes, geometric-coordinate errors and the scope of applying a constant correction. On the same common references, oracle repair of anchors carrying the old WI label has the largest HC category contribution; a separate geometry game assigns more of the repair contrast to center coordinates than sizes. Applying the unchanged erroneous delta outside the planned-anchor hull adds loss, but restricting it to the hull does not rescue full DINO and also discards useful correct-reference gains. These are overlapping controlled contrasts, not three independent causes or causal percentages.','',
 '## Frozen question and label provenance','',
 'The original32 exposed rollback queries, saved Before state, Source sensitivity, final-distribution MBR, four original planned positions and admitted DINO boxes are fixed. There is no new model, expert, training, Student update, pixel inspection or online replay. Final MBR remains the user’s fixed engineering component; its earlier T0 failure is unchanged. All diagnostic candidates were sealed before this run’s dense scoring. Only old sealed sparse GT entered oracle repair generation.','',
 'Old VISUAL_REVIEW labels are reused only after exact query/parent roster and panel-SHA binding: WI apparent wrong participant, CP partial/uncertain extent, MX merged instances, CI apparently consistent identity, UN uncertain. Six semantic-role-unknown queries have every slot conservatively overridden to UN while retaining the original codes. There are no new independent human labels. Existing PNG file bytes were read for SHA256 binding, but no PNG was decoded or newly visually inspected; “no new pixel review” does not mean zero binary file I/O. Identity was not inferred from IoU.','',
 '## Category-conditioned oracle repairs','',
 'DINO_common and GT_common share exactly43 HC and24 Vid available reference positions and the same scored support. For each query, every subset of categories present is replaced by its cached GT boxes; exact finite Shapley values allocate the GT_common−DINO_common contrast. The common-set filter itself uses GT availability and is not deployable. All16 parents/domain remain in primary category means, including absent-category zeros and no-reference no-ops. Category-present conditional means are secondary and use different denominators.','',
 '| HC old category | Common anchors / present parents | Full vIoU repair Shapley pp [95% CI] | Unobserved actual-decoder repair Shapley pp [95% CI] |','|---|---:|---:|---:|']
 for g in GROUPS:
  c=hc['category_oracle_repair_shapley']['v'][g];lines.append(f"| {g} | {c['reference_count']} / {c['present_parents']} | {fmt(c['cohort_contribution'])} | {fmt(hc['category_oracle_repair_shapley']['unobserved'][g]['cohort_contribution'])} |")
 lines+=['',
 'The HC total matched repair gap is+7.754662pp full vIoU and+11.955165pp unobserved sIoU. WI contributes+6.801663/+11.566492pp in this model-assisted-label game. The remaining categories have smaller positive and negative contributions; these are not causal fractions or independently certified identity effects. WI-box GT replacement jointly repairs identity and geometry for references carrying that tag; it does not isolate pure identity. Only five HC parents contain WI, and labels are not blind independent human annotations. Replacing only WI references, leaving every other DINO box unchanged, improves over DINO_common by+6.777871/+11.566516pp and yields+8.212862/+9.617499pp versus Base. Replacing only non-WI references yields+2.364199/−1.560320pp versus Base. These oracle interventions do not validate an automatic WI detector or filter.',
 '', 'Reference geometry is compatible with those old labels but does not recertify them. HC WI common-reference mean IoU is.043684 and target-size-normalized center error1.463254. CI has15 anchors across6 parents and mean IoU.772741, so apparently consistent identity still does not mean exact GT geometry. CP has only2 anchors in2 parents, mean area ratio.391549; MX has3 anchors in1 parent, area ratio1.641895. Their small or uncertain repair effects cannot establish that partial/merged extent error is unimportant. UN retains unknown-role and ambiguous cases, including negative cohort contributions.',
 '', 'Vid is context, not a second HC replication: category full-v contributions WI+1.437281, CP+.283173, MX+1.422806, CI+.085551, UN+4.050006pp; unobserved contributions+5.421643/+2.258653/+1.198197/+.256812/+7.537842. Only2/1/1/5/2 parents contain these categories. Tiny counts and old model-assisted uncertainty preclude a general identity-error ranking.',
 '', '## Center and size: a different exact game','',
 'Four hybrids hold common positions fixed: DINO center/DINO size, GT center/DINO size, DINO center/GT size and GT center/GT size. They use the same constant LS and global scope. HC center/size Shapley contributions are+6.093863/+1.660799pp full v, and+10.464651/+1.490514pp unobserved s. They sum to the same total matched gap as the category game, so the two decompositions must not be added.',
 '', 'GT-center-only gives HC+5.967710pp full v and+6.945691pp unobserved s versus Base; GT-size-only gives+1.534645/−2.028447pp. This coordinate intervention favors correcting center coordinates in this panel. Center is not a pure identity variable: partial boxes can shift centers, and wrong participants change both center and size. Size is not all extent error. Size also interacts with center: after centers are replaced by GT, correcting size adds+3.221943pp full v and+3.060458pp unobserved s. Size-alone changes unobserved s by−.079430pp versus DINO_common, so its weak standalone contrast does not mean size is useless. The result does not identify identity as the unique underlying cause or imply that scale never matters.',
 '', '## Where the same delta is applied','',
 'No delta is re-estimated when scope changes. Masks are global, the inclusive physical hull of the original four planned positions, or the original admitted decoder nodes. All three masks are defined without GT. Node-only applies the same global delta at admitted nodes; it is different from replacing those nodes with reference targets (the old LocalAnchor control). Benchmark interpolation near mask boundaries remains unchanged and contributes to full-v metrics.',
 '', '| HC reference / same delta | Global Δv / unobserved Δs pp | Planned-hull Δv / unobserved Δs pp | Admitted-node Δv / unobserved Δs pp |','|---|---:|---:|---:|']
 for ref in ['DINO_full','DINO_common','GT_common']:
  lines.append('| '+ref+' | '+' | '.join(f"{hc['scope'][ref][s]['v']['mean']:+.6f} / {hc['scope'][ref][s]['unobserved']['mean']:+.6f}"for s in ['global','planned_hull','admitted_nodes'])+' |')
 lines+=['',
 'For full DINO the HC unobserved loss decomposes additively as−.683930pp inside-hull contribution plus−1.991837pp outside-hull contribution =−2.675767pp. Each part divides by that query’s original total known unobserved count before parent averaging. Conditional inside-only mean−5.387585 and outside-only mean−3.436294 use different support denominators and must not be added. Restricting application to the hull removes outside loss but leaves negative full-DINO unobserved mean; the failure is not only extrapolation.',
 '', 'DINO_common hull-only has+.485930pp unobserved mean, but it uses the label-availability-selected reference set and is not an unlabeled improvement. For correct GT_common the global unobserved gain+10.006148 splits into+3.885960 inside contribution and+6.120188 outside contribution. A hard hull loses the latter benefit. The diagnostic does not choose or deploy a mask. Admitted-node-only has essentially zero actual-unobserved effect by construction, while benchmark interpolation can still change.',
 '', '## Limits of correct references, uncertainty and cost','',
 'Correct reference boxes do not make a constant LS fit exact at every anchor, nor optimize IoU directly. Residual losses can reflect limited constant expressiveness, averaging and chart-space/IoU mismatch; this diagnostic does not separately identify those remaining mechanisms. They must not all be called propagation failure. In saved HC examples, H003 unobserved DINO_common−34.692238 becomes GT_common−6.016689; H010−26.552276 becomes−21.826281; H005−70.001307 becomes+11.360462. These are illustrative cached-score cases, not new labels or selected evaluation parents. All original32 remain, with negative/neutral/unknown outcomes and saved-score leave-one ranges.',
 '', 'Source sensitivity, full and within-MBR endpoints, category-present counts, conditional reference quality, all parent signs, inside/outside supports and bootstrap CIs remain in RESULTS/DIAGNOSTICS. Reference-quality means weight anchors; primary repair effects weight parents. The10,000 fixed-seed parent intervals describe saved inherited trajectories and are not stream replay, multiplicity-adjusted confirmation or independent human-label validation.',
 '', f"Independent auditing reconstructs{audit['independent_candidate_output_checks']} candidate outputs (max error{audit['max_candidate_error']:.6g}), evaluates{audit['permutation_Shapley_games']} games by full permutations instead of the runner’s subset-weight formula, verifies{audit['old_matched_metric_checks']} old endpoint identities and{audit['scalar_aggregate_CI_quality_count_partition_checks']} scalar/aggregate/CI/quality/count/support checks. Category coordinate-error vectors also sum to the DINO-minus-GT delta independently. The largest inside/outside additive error is{audit['max_inside_outside_sum_error']:.6g}. All original runtime/cache/registry pins remain exact.",
 '', f"Cached candidate generation took{bar['CPU_wall_seconds']:.6f}s and dense scoring{completion['CPU_score_seconds']:.6f}s; this is diagnostic CPU cost, not fresh online latency. New GPU/model/expert/training/Student-update and new visual-label counts are0. The old visual panel was only byte-hashed for provenance.",
 '', 'The permitted conclusion is conditional: apparent wrong-participant references are a substantial oracle repair target in the old HC label game; center-coordinate repair is substantial in a separate geometric game; applying a wrong delta globally extends loss but hard restriction also removes valid gains. No causal percentage, new filter, chosen hybrid, dataset-specific recipe, production winner, OPD or automatic follow-up follows. Existing deployments and all earlier positive/negative results remain unchanged.',
 '', '- [All anonymous values, exact games, Shapley, scope effects, quality and CIs](RESULTS.json)',
 '- [Secondary contrasts, conditional counts and saved-score sensitivity](DIAGNOSTICS.json)',
 '- [Independent audit](ROOT_INDEPENDENT_AUDIT.json), [verification](VERIFICATION.json), [actual protocol/checkpoint hashes](ACTUAL_PROTOCOL.json)',
 '- [Aggregate figure](HC_DIAGNOSIS.svg), [locked protocol](../../../../protocols/stvg_error_state_hc_diagnosis32_v1.md)',
 '', 'Raw coordinates, sparse/dense GT, original query/parent text, media, q, weights and raw tensors are excluded from the public export. Old categorical labels and anonymous scalar quality measures retain their model-assisted uncertainty.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# HC sparse-error diagnosis cached32\n\n[Report](REPORT.md), [full anonymous results](RESULTS.json), [actual protocol](ACTUAL_PROTOCOL.json), [independent audit](ROOT_INDEPENDENT_AUDIT.json).\n\nCategory-conditioned oracle repair, center/size geometry and application scope are overlapping diagnostics, not additive causes or a chosen deployable filter. No new model/expert/training/visual labels or OPD.\n')
 files=['scripts/run_stvg_error_state_hc_diagnosis32_v1.py','scripts/audit_stvg_error_state_hc_diagnosis32_v1.py','scripts/report_stvg_error_state_hc_diagnosis32_v1.py','protocols/stvg_error_state_hc_diagnosis32_v1.md']
 files += [str((P/f).relative_to(ROOT))for f in ['RESULTS.json','ROOT_INDEPENDENT_AUDIT.json','ACTUAL_PROTOCOL.json','DIAGNOSTICS.json','VERIFICATION.json','HC_DIAGNOSIS.svg','REPORT.md','README.md']]
 put(D/'PUBLIC_FILES.json',dict(files=files));print(json.dumps(dict(files=len(files),bytes=sum((ROOT/f).stat().st_size for f in files))))
if __name__=='__main__':main()
