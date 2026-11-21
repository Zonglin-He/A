"""Run the pinned prior Qwen inference on the new sealed panel."""
import argparse
from pathlib import Path
from unittest.mock import patch

from cohort import OUT, PRIVATE, PROTOCOL, QWEN, read, read_panel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    args = parser.parse_args()
    panel = read_panel()
    assert read(PROTOCOL)['inference'] == read(QWEN / 'PROTOCOL.json')['inference']
    import infer

    with patch.multiple(infer, OUT=OUT, PRIV=PRIVATE, SPEC=PROTOCOL):
        infer.main(args.model_dir)
    assert read(OUT / 'model_seal.json')['files'] == 2 * len(panel)
