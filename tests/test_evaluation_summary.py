"""CPU-only synthetic checks: no dataset labels, video decoding or model inference."""
import json
import numpy as np
import pytest
from stvg import evaluation


def row(key,parent,value,exposure='unknown'):
    values={}
    for arm,delta in [('Source',0),('Spatial',.01),('Temporal',.02),('Full',.03),('WithoutOPD',.01)]:
        v=dict(v=value+delta,t=.5,s=.7,fixed_actual_sIoU=.6,
               observed_actual_sIoU=.8,unobserved_actual_sIoU=None,
               boundary_MAE=60.)
        values[arm]=evaluation.add_reporting_metrics(v,{'fps':30.})
    return dict(id=key,parent=parent,dataset='synthetic',values=values,
                development_exposure=exposure)


def test_query_main_table_is_separate_from_parent_average():
    rows=[row('one','a',.3,'current_development_parent'),
          row('two','a',.5,'current_development_parent'),
          row('three','b',.9,'not_current_development_parent')]
    z=evaluation.summarize_rows(rows)['synthetic']
    assert z['queries']==3 and z['parents']==2
    assert z['query_main_table']['Source']['m_vIoU']==pytest.approx(1.7/3)
    assert z['means']['Source']['v']==pytest.approx(.65)
    assert z['parent_main_table']['Source']['m_vIoU']==pytest.approx(.65)
    assert z['query_main_table']['Source']['vIoU_ge_03']==1
    assert z['query_main_table']['Source']['vIoU_ge_05']==pytest.approx(2/3)
    assert z['query_main_table']['Source']['m_sIoU']==pytest.approx(.7)
    assert z['query_main_table']['Source']['m_sIoU_fixed_actual']==.6
    assert z['query_main_table']['Source']['vIoU@0.3']==pytest.approx(2/3)
    assert z['query_main_table']['Source']['vIoU@0.5']==pytest.approx(1/3)
    assert z['query_main_table']['Source']['m_sIoU_unobserved_actual']is None
    assert z['query_valid_counts']['Source']['unobserved_actual_sIoU']==0
    assert z['paired']['Full-Source']['mean_pp']==pytest.approx(3.)
    assert z['paired']['Full-WithoutOPD']['CI95_pp']==pytest.approx([2.,2.])
    assert z['development_strata']['current_development_parent']['queries']==2
    assert z['development_strata']['not_current_development_parent']['parents']==1


def test_reporting_thresholds_and_fps_are_explicit():
    values=dict(v=.3,t=0.,boundary_MAE=30.,fixed_actual_sIoU=.2,
                observed_actual_sIoU=.1,unobserved_actual_sIoU=.3)
    z=evaluation.add_reporting_metrics(values,{'fps':15.})
    assert z['vIoU_ge_03']==1 and z['vIoU_ge_05']==0
    assert z['vIoU_at_03']==0 and z['vIoU_at_05']==0
    assert z['tIoU_eq_0']==1 and z['tIoU_ge_05']==0
    assert z['boundary_MAE_seconds']==2.
    assert z['boundary_MAE']==30.  # Existing physical-frame field stays unchanged.
    for invalid in [None,0,-1,True,'30',float('nan'),float('inf')]:
        assert evaluation.add_reporting_metrics(values,{'fps':invalid})['boundary_MAE_seconds']is None
    assert evaluation.add_reporting_metrics(values,{})['boundary_MAE_seconds']is None
    values.update(v=.5,t=.5)
    z=evaluation.add_reporting_metrics(values,{'fps':30.})
    assert z['vIoU_ge_05']==1 and z['tIoU_ge_05']==1
    assert z['vIoU_at_03']==1 and z['vIoU_at_05']==0
    values['v']=float(np.nextafter(.5,1.))
    assert evaluation.add_reporting_metrics(values,{'fps':30.})['vIoU_at_05']==1


def test_all_seals_verified_before_any_label_read(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.jsonl'
    records=[dict(id='a'),dict(id='b')]
    manifest.write_text(''.join(json.dumps(r)+'\n'for r in records))
    root=tmp_path/'predictions';root.mkdir()
    files={}
    for key in ['a','b']:
        p=root/(key+'.npz');p.write_bytes(b'original')
        files[key]=dict(path=p.name,sha256=evaluation.sha(p))
    (root/'sealed.json').write_text(json.dumps(dict(manifest_sha256=evaluation.sha(manifest),queries=2,files=files)))
    (root/'b.npz').write_bytes(b'changed later prediction')
    labels=tmp_path/'must_not_read_labels.jsonl'
    real=evaluation.read_rows
    def guarded(path):
        assert path!=labels,'Labels opened before every prediction was verified'
        return real(path)
    monkeypatch.setattr(evaluation,'read_rows',guarded)
    with pytest.raises(ValueError,match='content does not match'):
        evaluation.score(manifest,labels,root,tmp_path/'metrics.json')


def test_cohort_receipt_uses_label_free_parent_ids(tmp_path):
    def item(key,parent,fps):
        return dict(id=key,dataset='hc2',metadata=dict(source=parent,fps=fps))
    dev=tmp_path/'development.jsonl';dev.write_text(json.dumps(item('old','a',30.))+'\n')
    manifest=tmp_path/'cohort.jsonl';rows=[item('one','a',30.),item('two','a',None),item('three','b',15.)]
    manifest.write_text(''.join(json.dumps(r)+'\n'for r in rows))
    receipt,known=evaluation.cohort_provenance(rows,manifest,dev)
    assert receipt['manifest_sha256']==evaluation.sha(manifest)
    assert receipt['development_manifest_sha256']==evaluation.sha(dev)
    assert known=={'hc2':{'a'}}
    ds=receipt['datasets']['hc2']
    assert ds==dict(queries=3,parents=2,current_development_parents=1,
                   current_development_queries=2,missing_or_invalid_fps_queries=1)
    receipt,known=evaluation.cohort_provenance(rows,manifest,tmp_path/'missing.jsonl')
    assert not receipt['development_manifest_available']
    assert receipt['datasets']['hc2']['current_development_parents']is None
    assert known=={}
