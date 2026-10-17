"""Root-only numerical audit correction; no Student updates or prediction recomputation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_semantic_preference_student32_v1 as study,audit_stvg_spatial_lite_fit002 as audit
from scripts.audit_stvg_fixed_support_adam002 import adam_audit
if __name__=='__main__':
    for p,h in study.read(study.BASE/'RECOVERY_RUNTIME002.json')['pins'].items():assert study.sha(ROOT/p)==h,p
    audit._adam_audit=adam_audit;study.root(sys.argv[1])
