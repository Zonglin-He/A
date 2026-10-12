"""Prepared CPU review binding correction; actual Direct uses its own audit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,sha
import scripts.review_stvg_opd_common_state_postseal_v1 as original
REC=original.REC/'preGT_objective_dispatch002'


def run():
    for p,h in read(REC/'REVISION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    import scripts.stvg_opd_design_ablation_common_v1 as common
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit as direct
    qualified=common.audit
    def dispatch(fit,expert):
        return direct(fit,expert) if fit.get('direct_objective') else qualified(fit,expert)
    common.audit=dispatch
    original.run()


if __name__=='__main__':run()
