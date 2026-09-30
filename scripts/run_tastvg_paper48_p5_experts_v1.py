"""Same specialist implementations, new HC2 observations; serial GPU owner."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_p5_common_v1 import BASE,verify

def run(stage):
    # The reused worker imports decode lazily; bind HC's official timing only here.
    from vg_tta import exact_frame_decode_audit_v2 as decoder_binding
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
    decoder_binding.decode=decode
    from scripts import run_tastvg_paper48_experts_v1 as w
    w.OUT=BASE/'experts';w.OLD=BASE/'no_external_cache';w.reuse_receipts=lambda *args:None
    w.verify=lambda:verify(expert=True)
    w.run(stage)
if __name__=='__main__':run(sys.argv[1])
