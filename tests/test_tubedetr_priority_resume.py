import ast
import inspect
import pytest
from tools.tubedetr_spatial_search import full, priority_resume as priority


def test_priority_keeps_frozen_gpu_numerics_and_qualification():
    source = inspect.getsource(full.gpu)
    tree = priority.authorized_gpu_tree()
    expected = ast.parse(source.replace(priority.ORDER_ASSERT, '_authorize()'))
    assert ast.dump(tree, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert 'features_Source_raw_selected_output_bitexact' in ast.unparse(tree)
    assert 'flock' in ast.unparse(tree)


def test_priority_refuses_missing_or_duplicate_order_gate():
    source = inspect.getsource(full.gpu)
    with pytest.raises(ValueError):
        priority.authorized_gpu_tree(source.replace(priority.ORDER_ASSERT, 'pass'))
    with pytest.raises(ValueError):
        priority.authorized_gpu_tree(source.replace(priority.ORDER_ASSERT, priority.ORDER_ASSERT + '\n    ' + priority.ORDER_ASSERT))


def test_priority_rejects_active_llava(monkeypatch):
    monkeypatch.setattr(priority, 'checked', lambda: None)
    monkeypatch.setattr(priority, 'active_llava', lambda: [123])
    with pytest.raises(AssertionError, match='LLaVA must remain paused'):
        priority._authorize()
