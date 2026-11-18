"""Recreate the private M3ED calibration bridge using only public HDF5 calibration datasets."""
from functools import lru_cache
from io import RawIOBase
import json
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 32768


class RangedHDF(RawIOBase):
    """Provide h5py seekable byte ranges without downloading a recording."""
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        head = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        head.raise_for_status()
        if head.status_code != 206:
            raise ValueError("HDF5 provider did not support range requests")
        self.length = int(head.headers["Content-Range"].split("/")[-1])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.length)[whence] + offset
        return self.pos

    @lru_cache(maxsize=256)
    def block(self, index):
        begin = index*BLOCK
        end = min(self.length, begin + BLOCK) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={begin}-{end}"}, timeout=30)
        r.raise_for_status()
        if r.status_code != 206 or r.headers["Content-Range"] != f"bytes {begin}-{end}/{self.length}":
            raise ValueError("remote calibration byte-range mismatch")
        return r.content

    def readinto(self, dest):
        start = self.pos
        size = min(len(dest), self.length - start)
        if size <= 0:
            return 0
        blocks = b"".join(self.block(i) for i in range(start//BLOCK, (start+size+BLOCK-1)//BLOCK))
        dest[:size] = blocks[start%BLOCK:start%BLOCK+size]
        self.pos += size
        return size

    def close(self):
        self.block.cache_clear()
        self.session.close()
        super().close()


def acquire():
    from heldout157 import REG, PINS, digest, require, read, CALIBRATION_HASH, CALIBRATION
    import hashlib
    require(digest(REG) == PINS[REG], "unrecognized ordinary manifest")
    scenes = {(e["target"], e["row"]["scene"]) for e in read(REG) if e["target"] in ("drone", "quad")}
    records = {}
    for target, scene in sorted(scenes):
        name = ("falcon" if target == "drone" else "spot") + "_" + scene.lower()
        url = f"https://m3ed-dist.s3.amazonaws.com/processed/{name}/{name}_data.h5"
        with RangedHDF(url) as remote, h5py.File(remote, "r") as h5:
            event_from_lidar = h5["/ouster/calib/T_to_prophesee_left"][()]
            event_from_rgb = h5["/ovc/rgb/calib/T_to_prophesee_left"][()]
            metadata = json.loads(h5["/ouster/metadata"][()])
        records[scene] = dict(T_cam_lidar=(np.linalg.inv(event_from_rgb) @ event_from_lidar).tolist(),
                              metadata=metadata, url_seq=name)
    payload = json.dumps(records, sort_keys=True).encode()
    require(hashlib.sha256(payload).hexdigest() == CALIBRATION_HASH, "primary calibration changed")
    if CALIBRATION.exists():
        require(digest(CALIBRATION) == CALIBRATION_HASH, "existing reference calibration differs")
    else:
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        with CALIBRATION.open("xb") as f:
            f.write(payload)


if __name__ == "__main__":
    acquire()
