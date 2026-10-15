"""Use the established explicit public allowlist staging and index verification."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_soft_ep_v1 as publisher
from scripts.stvg_native_instance_common_v1 import read
BASE=ROOT/'artifacts/stvg_temporal_role_alignment_v1/revision002'
OUT=ROOT/'results/stvg_temporal_role_alignment/2026-10-15'
SOURCE=['scripts/'+x for x in ['stvg_temporal_role_alignment002.py','run_stvg_temporal_role_alignment002.py','review_stvg_temporal_role_alignment002.py','measure_stvg_temporal_role_alignment002.py','launch_stvg_temporal_role_alignment002.py','continue_stvg_temporal_role_alignment002.py','stvg_soft_ep_coordinate_common_v1.py','stvg_evidence_temporal_fit_v1.py','stvg_table1_temporal_adapter_v1.py','stvg_temporal_caption_query_v1.py','report_stvg_temporal_role_alignment_v1.py','publish_stvg_temporal_role_alignment_v1.py']]
def paths():
 assert read(BASE/'ROOT_INDEPENDENT_SCALAR_AUDIT002.json')['status']=='pass'
 assert read(OUT/'REPORT_RECEIPT.json')['status']=='complete'
 return set(SOURCE+['docs/STVG_TEMPORAL_ROLE_ALIGNMENT_V1.md','protocols/stvg_temporal_role_alignment_v1.md']+[str((OUT/x).relative_to(ROOT))for x in ['ANONYMOUS_ROWS.json','ANONYMOUS_SUMMARIES.json','ANONYMOUS_RANKING.json','SOURCE_BINDINGS.json','REPORT_RECEIPT.json']])
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['stage','verify-stage']);args=a.parse_args()
 publisher.BASE=BASE;publisher.expected_paths=paths
 if args.mode=='stage':publisher.stage(publisher.DEFAULT_CHECKOUT,None,False)
 else:publisher.verify_stage(publisher.DEFAULT_CHECKOUT)
if __name__=='__main__':main()
