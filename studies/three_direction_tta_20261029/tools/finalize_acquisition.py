"""One-shot completion record for the already-authorized archive transfers.

No model, annotation, media or score access; observes only acquisition receipts.
"""
import json, time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def main():
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        avis=read(ROOT/'avis/full_archive_verification_state.json')
        great=read(ROOT/'affordance/DATASET_SPLIT_VALIDATION_RECEIPT.json')
        three=read(ROOT/'crossplatform/full_dataset_verified.json')
        status={'recorded_utc':datetime.now(timezone.utc).isoformat(),
                'avis':avis,'great':great,'3eed':three,
                'scope':'One-shot integrity record for existing transfers only; no new model/data scoring or auto-publication'}
        if avis.get('state')=='completed' and great.get('state')=='complete':
            assert avis['gzip_full_stream_crc_and_size_verified'] and avis['original_stream_gt_matches_recovered_tail']
            assert great['all_archive_members_crc_valid'] and great['all_extracted_member_sizes_match']
            status['state']='all_selected_direction_archives_and_splits_verified'
            save(ROOT/'research/ACQUISITION_FINAL.json',status)
            current=read(ROOT/'research/STATE.json')
            current['directions']['avis']['dataset']='full_archive_complete_SHA_recorded_gzip_CRC_and_original_GT_crosscheck_pass'
            current['directions']['affordance']['dataset']='full_Unseen_aff_archive_CRC_and_split_validation_pass_materialized'
            current['acquisition_final_receipt']='ACQUISITION_FINAL.json'
            save(ROOT/'research/STATE.json',current)
            print(status['state'],flush=True);return
        if avis.get('state')=='failed' or (ROOT/'affordance/DATASET_SPLIT_VALIDATION_FAILURE.json').exists():
            status['state']='integrity_failure_requires_investigation';save(ROOT/'research/ACQUISITION_FINAL.json',status)
            raise RuntimeError(status['state'])
        time.sleep(30)
    save(ROOT/'research/ACQUISITION_FINAL.json',{'state':'waiting_timeout_no_completion_claim','recorded_utc':datetime.now(timezone.utc).isoformat()})
    raise TimeoutError('Archive verification not complete within6hours; no full completion claim')

if __name__=='__main__':main()
