"""Root language-input contract clarification; no new model or GT scoring."""
from pathlib import Path
import sys,inspect,json,time,hashlib
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/temporal_query_semantics_v1'))
from common import *
def main():
 validate();guard();torch.set_num_threads(1);assert(D/'ROOT_completion.json').exists();oldroot=read(D/'ROOT_completion.json');audit=read(D/'audit_receipt.json');assert audit['report_sha256']==sha(P/'report.json');cohort=read(D/'inventory_v3.json')['selected'];seal=read(D/'sealed.json');tests=[]
 # There is no admissible direct text argument. Test the real API rather than
 # fabricate a language-aware adapter that silently ignores a new input.
 for pair in cohort:
  for own,partner in [(pair['a'],pair['b']),(pair['b'],pair['a'])]:
   e=seal['files'][own['id']];assert sha(D/e['path'])==e['sha256']
   with np.load(D/e['path'])as z:
    try:one_step(z['input_F'],z['input_q0'],kappa=0,beta=1,caption=partner['caption'])
    except TypeError as exc:assert 'caption'in str(exc)and'unexpected keyword'in str(exc);tests.append(dict(query=own['id'],expected_unsupported_caption_keyword_rejected=True,existing_arrays_not_changed=True))
    else:raise AssertionError('Unexpected new text input accepted')
 assert len(tests)==64;atomic(D/'actual_language_API_probe.json',dict(signature=str(inspect.signature(one_step)),tests=tests,calls=64,expected_TypeError_count=64,actual_model_forward=0,new_Source=0,new_expert=0,new_GT_score=0,new_GPU=0,worker=identity(),time=time.time()));atomic(P/'language_API_contract.json',dict(status='actual64unsupported_caption_keyword_calls_confirm_no_direct_text_interface',signature=str(inspect.signature(one_step)),queries=64,expected_TypeError_count=64,caption_probe_details_private=True,derived_arrays_previous_field_names_clarified={'text_metadata_q_exact_percent':'identical tensor-input repeat posterior exactness; no text was passed in those numerical replays','text_metadata_L_exact_percent':'identical tensor-input repeat L exactness; no text was passed in those numerical replays'},new_endtoend_action_text_Source_or_expert_forward_performed=False,existing_natural_action_queries_actual_Source_predictions_reused=True,new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0))
 text=(P/'ANALYSIS.md').read_text();a=text.index('## Direct language-input contract');b=text.index('## Scope and validity',a);old_section=text[a:b];atomic(D/'preserved/root_contract_report_v1/diagnosis.json',dict(reason='Frozen derive caption argument was unused and defaulted to empty; fields named text_metadata_* measure identical-input repeat exactness, not an executed caption-swap forward. Clarify the report and actually test the real API rejection. All predictions/scores/99CI/audit hashes unchanged.',new_source=0,new_expert=0,new_model=0,new_GT_score=0));(D/'preserved/root_contract_report_v1/original_section.txt').write_text(old_section)
 replacement='''## Direct language-input contract: scope correction

Actual frozen one_step signature: '''+str(inspect.signature(one_step))+''' . The update has no text argument. The sealed arrays labeled text_metadata_* were identical-F/q0 repeat calls: their caption default was unused. Those exactness rates must NOT be described as a new caption-substitution response experiment. They are deterministic replay controls only.

ROOT separately executed64actual calls to the real one_step API using the partner action query as a caption keyword. All64 correctly raised unsupported-keyword TypeError before any numerical/model computation. This confirms the absence of a direct text entry, not a test of whether Source changes under an edited action description. We do not fabricate a text adapter or claim new end-to-end action-text counterfactuals. The zero-feature control exactly reproduces q0.

The32real query pairs already have actual query-conditioned Source/expert outputs in the audited caches. Their natural query differences can enter via Source priors/boxes/Eregions/pooled features, but attributes and eight planned positions can change too. The feature/prior factorial probes test dependence on those upstream states conditional on fixed cached inputs. They do not independently prove action semantics inside Eq6-8.

'''
 text=text[:a]+replacement+text[b:];s=read(P/'report.json')['summary']['statistics'];intro='''## ROOT conclusion

On this prespecified32-video same-actor/object different-action cohort, matched JointPrior tIoU20.969638% becomes19.898830% after the unchanged E-ROI visual update; difference−1.070808pp,95%videoCI[−3.291101,+.888341]. Correct-query versus swapped-GT assignment margin+.049714ppCI[−4.577302,+3.290746], with17positive/11negative/4ties; no resolved action-specific matching advantage. Own versus partner features at the same own prior difference−.186300ppCI[−.751373,+.256350]; own versus partner evidence−.253634ppCI[−1.601092,+1.022079], also unresolved. Both events reach tIoU>=.3 in0/32pairs after update; this small, deliberately difficult, label-selected diagnostic is not a population zero-capability claim.

Swapping prior changes55/64intervals(85.9375%), swapping features33/64(51.5625%). Under a common symmetric prior,17/32pairs output identical intervals and mean pair interval overlap rises to.945470. Exclusive-action evidence contrast on27scorablepairs is+.018156CI[−.049772,+.085046]. These are consistent with material prior conditioning and do not establish independent action-semantic recognition; unequal intervention strength prevents a quantitative causal contribution ranking. Source and the unchanged method can remain useful on broader benchmarks even though this targeted cohort supplies no positive independent semantic evidence.

'''
 text=text.replace('## Scope and validity',intro+'## Scope and validity');(P/'ANALYSIS.md').write_text(text)
 oldroot.update(status='same_actor_object_action_cohort_scored_audited_ROOTinterpreted_language_contract_corrected_PENDING_publication',analysis_sha256=sha(P/'ANALYSIS.md'),original_ROOT_analysis_sha256=oldroot['analysis_sha256'],contract_probe_sha256=sha(P/'language_API_contract.json'),contract_probe_code_sha256=sha(Path(__file__)),original_scientific_arrays_and_report_unchanged=True,no_new_endtoend_edited_action_text_forward=True,time=time.time());atomic(D/'ROOT_completion_v2.json',oldroot);atomic(P/'completion_scalars.json',oldroot);print(json.dumps(dict(actual_caption_keyword_rejections=64,arrays_report_audit_unchanged=True,primary_visual_gain=s['visual_gain_t_pp'],query_event_assignment_margin=s['CorrectedROI_assignment_margin_pp'])))
if __name__=='__main__':main()
