"""Explicit allowlist staging using the project's established public-export workflow."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_soft_ep_v1 as publisher

BASE=ROOT/'artifacts/stvg_component_ablation128_v1'
OUT=ROOT/'results/stvg_coordinate_component128/2026-10-16'


def paths():
    audit=json.loads((OUT/'ROOT_PUBLIC_SCALAR_AUDIT.json').read_text())
    assert audit['status']=='pass' and audit['rows_checked']==52864
    result={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}
    result.update(['docs/STVG_COORDINATE_COMPONENT128_V1.md','scripts/export_stvg_coordinate_component128_v1.py','scripts/audit_stvg_coordinate_component128_public_v1.py','scripts/publish_stvg_coordinate_component128_v1.py','scripts/read_stvg_native_final_v1.py','scripts/read_stvg_coordinate_sensitivity_v1.py','scripts/read_stvg_coordinate_component_progress_v1.py'])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','verify']);a=parser.parse_args()
    publisher.BASE=BASE;publisher.expected_paths=paths
    if a.mode=='stage':publisher.stage(publisher.DEFAULT_CHECKOUT,None,False)
    else:publisher.verify_stage(publisher.DEFAULT_CHECKOUT)


if __name__=='__main__':main()
