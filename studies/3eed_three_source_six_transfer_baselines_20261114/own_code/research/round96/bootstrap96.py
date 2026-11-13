from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round96';now=datetime.datetime.now(datetime.timezone.utc).isoformat();rd=lambda p:json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
g=rd(R/'research/STATE.json');dump(H/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json',g)
s={'state':'R96_AUTHORIZED_THREE_SOURCE_SIX_TRANSFER_BASELINES','execution_open':True,'manual_authorization':True,'automations_remain_off':True,'GPU_stage_authorized':True,'CPU_scoring_authorized':True,'source_training_authorized':True,'FINAL_access_authorized':False,'sources_to_train':['drone','quad'],'reused_source':'Waymo-only epoch100 round6','primary_epoch':100,'serial_GPU_only':True,'current_stage':'official_asset_and_data_audit','started_utc':now,'crossplatform_TTA_claim':False};dump(H/'STATE.json',s)
g.update(current_round='R96',manual_active_state='round96/STATE.json',run_state=s['state'],current_stage=s['current_stage'],updated_utc=now,next_action='audit then train drone100 and quad100 sequentially');g['directions']['crossplatform'].update(latest_round='R96',active_state='round96/STATE.json',active_experiment='three_source_six_transfer_official_baselines',state=s['state']);dump(R/'research/STATE.json',g)
