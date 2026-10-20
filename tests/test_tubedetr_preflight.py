"""HC output indices must not be checked against native frame counts."""
from types import SimpleNamespace

import pytest

from tools import preflight_tubedetr_media as preflight


def test_hc_timestamp_duplicate_covers_last_requested_index(monkeypatch):
    metadata = dict(video_path='synthetic.mkv', video_sha256='fixed', width=2, height=2)
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr=b'', stdout=
            b'frame=599\ntotal_size=7188\nprogress=continue\n'
            b'frame=600\ntotal_size=7200\ndup_frames=1\ndrop_frames=0\nprogress=end\n')

    monkeypatch.setattr(preflight.subprocess, 'run', execute)
    output = preflight.hc_output_support('ffmpeg', metadata)
    stream = dict(width=2, height=2, nb_read_frames='599')
    preflight.validate_support(metadata, 599, stream, output['decoded_frames'])
    assert output['decoded_frames'] == 600 and output['duplicated_frames'] == 1
    assert '-vsync' not in calls[0] and '-fps_mode' not in calls[0]
    with pytest.raises(ValueError, match='physical support mismatch'):
        preflight.validate_support(metadata, 599, stream, 599)
    with pytest.raises(ValueError, match='physical support mismatch'):
        preflight.validate_support(metadata, 600, stream, 600)
    with pytest.raises(ValueError, match='physical support mismatch'):
        preflight.validate_support(metadata, 599, dict(width=3, height=2), 600)


@pytest.mark.parametrize('returncode,progress', [
    (1, b'frame=600\ntotal_size=7200\nprogress=end\n'),
    (0, b'frame=600\ntotal_size=7199\nprogress=end\n'),
    (0, b'frame=600\ntotal_size=7200\nprogress=continue\n'),
])
def test_hc_failed_truncated_or_incomplete_output_is_rejected(monkeypatch, returncode, progress):
    monkeypatch.setattr(preflight.subprocess, 'run', lambda *a, **kw:
        SimpleNamespace(returncode=returncode, stdout=progress, stderr=b'diagnostic'))
    with pytest.raises(ValueError, match='HC default RGB output failed'):
        preflight.hc_output_support('ffmpeg', dict(video_path='synthetic.mkv', width=2, height=2))
