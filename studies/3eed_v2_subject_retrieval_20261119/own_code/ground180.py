"""Experimental A256 subject retrieval at a fixed or source-selected K."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from retrieval180 import K_VALUES, RetrievalGrounder
from source180 import PLATFORMS, calibrated, prepare_device
from features import WAYMO_VIEWS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,choices=PLATFORMS)
    parser.add_argument('--target',required=True,choices=PLATFORMS)
    parser.add_argument('--points',required=True,type=Path)
    parser.add_argument('--rgb',required=True,type=Path,help='image dimensions for the frozen frontend')
    parser.add_argument('--calibration',required=True,type=Path)
    parser.add_argument('--text',required=True)
    parser.add_argument('--waymo-view',choices=WAYMO_VIEWS,default='F')
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--k',type=int,choices=K_VALUES)
    parser.add_argument('--retrieval',choices=('native','trained'),default='trained')
    args=vars(parser.parse_args())
    calibrated()
    prepare_device()
    with redirect_stdout(sys.stderr):
        model=RetrievalGrounder(args.pop('source'),asset_root=args.pop('asset_root'))
        output=model.predict(**args)
    print(json.dumps(output,allow_nan=False))


if __name__=='__main__':
    main()
