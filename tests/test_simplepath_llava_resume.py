from tools.simplepath_llava.resume import stages


def test_resume_keeps_full_order_and_never_reschedules_development():
    p = dict(dataset_order=['hc2', 'vidstg'], full_order=['vidstg', 'hc2'])
    assert stages(p) == [('gpu', 'vidstg'), ('score', 'vidstg'), ('gpu', 'hc2'), ('score', 'hc2')]
