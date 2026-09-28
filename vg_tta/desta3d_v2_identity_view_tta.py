"""Single view substitution; all calibration objective terms remain active."""
from vg_tta.desta3d_v2_tta_pilot import adapt


def adapt_identity_view(adapter, fields, *, moments):
    result = adapt(adapter, fields, fields, interface='calibration', alignment=.01,
                   moments=moments, steps=3, lr=1e-5)
    result.update(student_view='observed_identity', output_anchor=False,
                  sourcefit_reset_required=True)
    return result
