"""Preserve old report source, disclose the repaired Caption-matched factorial."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import FIXED,verify
def main():
    from scripts import report_stvg_factorial_native256_v1 as r,audit_stvg_factorial_readout_v1 as audit
    audit.BASE=FIXED;audit.verify=verify
    src=inspect.getsource(r.main).replace('stvg_factorial_native256_v1/2026-10-17/factorial256','stvg_factorial_native256_captionfix002/2026-10-17/factorial256').replace('STVG_NATIVE_FACTORIAL256_001.md','STVG_NATIVE_FACTORIAL_CAPTIONFIX002.md')
    src=src.replace('Existing Frozen, Temporal and Adapted-clock Full outputs are reused exactly.','Frozen, Temporal and Full-A are reused exactly. Full-B uses the old221 complete outputs plus35 actual Native-B full outputs only after exact same-clock E1 jobs, Source, Temporal interval and recipe checks. Historical original Full-B remains a separate row; the new comparison explicitly repairs Caption acquisition on both clocks.')
    src=src.replace("'publish_stvg_factorial_native256_v1.py'):","'publish_stvg_factorial_native256_v1.py','stvg_spatial_caption_repair_v1.py','continue_stvg_captionfix_native002.py','score_stvg_factorial_captionfix002.py','report_stvg_factorial_captionfix002.py','publish_stvg_factorial_captionfix002.py'):")
    ns=dict(r.__dict__);ns.update(BASE=FIXED,verify=verify);exec(compile(src,__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()
