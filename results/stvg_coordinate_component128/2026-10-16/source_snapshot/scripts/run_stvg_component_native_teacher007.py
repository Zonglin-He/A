"""Original hot R1 worker, isolated Native input namespace; no protocol edits."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_component_native007_common import BASE, verify

if __name__ == '__main__':
    verify()
    sys.argv = ['teacher', 'unused']
    from scripts import run_stvg_evidence_teacher_v1 as worker
    worker.BASE = BASE
    worker.main()
