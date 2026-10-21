"""Spatial readout denominators and constrained follow-up selection."""
from tools.analyze_tubedetr_spatial import joint_outcomes,signed_changes,pareto_front
from tools.tubedetr_vid_spatial_followup import choose


def test_missing_observed_gt_is_not_zero_filled_or_counted_in_joint():
    def row(obs,unobs,s,v):
        source=dict(observed_actual_sIoU=obs,unobserved_actual_sIoU=.5,m_sIoU=.5,m_vIoU=.2)
        full=dict(observed_actual_sIoU=None if obs is None else obs+.1,
                  unobserved_actual_sIoU=unobs,m_sIoU=s,m_vIoU=v)
        return dict(values=dict(Temporal=source,Full=full))
    rows=[row(.5,.4,.4,.3),row(None,.6,.6,.3)]
    j=joint_outcomes(rows)
    assert j['queries']==2 and j['paired_observed_unobserved_queries']==1
    assert j['observed_up_unobserved_down']==1
    d=signed_changes(rows,'Full','Temporal','observed_actual_sIoU')
    assert d['queries']==1 and abs(d['mean_pp']-10)<1e-12


def test_followup_rejects_spatial_gain_with_lost_volume_or_unobserved_quality():
    def value(s,v,u,n=60):
        return dict(params=dict(steps=n),means=dict(Full=dict(m_sIoU=s,m_vIoU=v,unobserved_actual_sIoU=u)))
    p=dict(reference=value(.4,.2,.4),volume_lost=value(.8,.199,.8),unobserved_lost=value(.8,.3,.399),
           good=value(.41,.2,.4),more_volume_less_space=value(.405,.3,.5))
    assert choose(p,'reference')=='good'
    assert choose({'reference':p['reference'],'bad':p['volume_lost']},'reference')=='reference'


def test_pareto_front_is_descriptive_and_retains_volume_spatial_tradeoffs():
    def v(volume,space,unobs):return dict(means=dict(Full=dict(m_vIoU=volume,m_sIoU=space,unobserved_actual_sIoU=unobs)))
    p=dict(volume=v(.3,.4,.4),space=v(.2,.5,.5),dominated=v(.1,.3,.3))
    assert pareto_front(p)==['space','volume']
