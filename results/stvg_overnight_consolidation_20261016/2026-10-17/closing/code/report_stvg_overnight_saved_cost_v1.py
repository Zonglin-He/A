"""Matched fixed32 saved cost readout, with missing instrumentation explicit."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
def main():
    c.verify();roster=c.read(c.BASE/'global_B/ROSTER_LOCK.json')['records'];full=c.read(c.CORE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(full,1)};chosen={r['probe_alias'] for r in roster};keys={aliases[x] for x in chosen};rows=[]
    for arm in ('Full_E1','A','B'):
        profile=c.read(ROOT/'artifacts/stvg_pipeline_speed_v1/student_profiles'/f'{arm}.json');pp={r['case_id']:r for r in profile['rows']};usage={x['case_id']:x for x in c.score(arm)['usage']};requests=[]
        for j in c.read(c.CORE/'TEACHER_JOB_LOCK.json')['records']:
            if j['case_id'] not in chosen:continue
            d=c.read(c.CORE/'teacher'/arm/(j['job_id']+'.json'));trace=d.get('sequence_readback') or {};requests.append(dict(case_id=aliases[j['case_id']],slot=j['center_slot'],new_teacher_call=d['new_teacher_call'],new_vision_call=d['new_vision_call'],wall_seconds=d['wall_seconds'],prefill_seconds=trace.get('prefill_seconds'),generation_forward_seconds=trace.get('generation_forward_seconds'),visual_tokens=d['visual_token_count'],output_count=d['output_count']))
        picked=[]
        for r in roster:
            key=aliases[r['probe_alias']];d=pp.get(key,pp.get(r['probe_alias']))
            if d is None:
                assert not usage[r['probe_alias']]['actual_fit'];picked.append(dict(case_id=key,dataset=r['dataset'],actual_fit=False,readout='exact already-computed Source/Temporal reuse; no new spatial fit or prefix restore',shared_baseline_cost_not_zero=True));continue
            picked.append({**{k:v for k,v in d.items() if k in ('dataset','cache_hit','source_decode_seconds','source_forward_seconds','source_state_hash_seconds','model_load_seconds','restore_seconds','capture_wall_seconds','final_readout_seconds','readout','feature_store_seconds')},'case_id':key,'actual_fit':True})
        assert len(picked)==32;rows.append(dict(arm=arm,student_source_and_readout=picked,teacher_requests=requests,cold_Source_captures=sum(r.get('cache_hit') is False for r in picked),warm_prefix_restores=sum(r.get('cache_hit') is True for r in picked),teacher_new_vision_calls=sum(r['new_vision_call'] for r in requests),teacher_new_generations=sum(r['new_teacher_call'] for r in requests)))
    out=c.BASE/'cost';out.mkdir(exist_ok=False);c.write(out/'SAVED_MATCHED32_COST.json',dict(status='complete_saved_instrumentation_only',count=32,records=rows,GT_read=False,cohort_prelocked_before_readout=True,actual_same_parent_costs=True,shared_candidate_and_Source_preparation_cost_is_not_zero=True,missing_components=['historical per-query candidate prepare and uninstrumented token-selection/solver subdivisions cannot be retrospectively invented','warm A/B reuse is not a matched cold-deployment latency trial'],end_to_end_15percent_savings_claim=False));c.write(out/'CPU_COMPLETION.json',dict(status='complete',count=32,CPU_only=True,GT_read=False,result_sha256=c.sha(out/'SAVED_MATCHED32_COST.json')))
if __name__=='__main__':main()
