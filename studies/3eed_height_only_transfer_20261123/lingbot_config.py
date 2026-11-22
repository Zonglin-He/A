"""Retrieve the published geometry backbone configuration from its checkpoint ZIP."""

import argparse
from collections import OrderedDict
from hashlib import sha256
import io
import json
from pathlib import Path
import pickle
from urllib.request import Request, urlopen
import zipfile


REV = "d1b06f7538d4b56387412799c528542c795df169"
URL = f"https://huggingface.co/robbyant/lingbot-depth-postrain-dc-vitl14/resolve/{REV}/model.pt"
SIZE = 1284841739


class CheckpointIndex(io.RawIOBase):
    def __init__(self):
        self.offset = 0

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = (0, self.offset, SIZE)[whence] + offset
        if self.offset < 0:
            raise OSError("Negative checkpoint offset")
        return self.offset

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, count=-1):
        limit = SIZE - self.offset if count < 0 else min(count, SIZE - self.offset)
        if not limit:
            return b""
        assert 0 < limit < 4*2**20
        last = self.offset + limit - 1
        req = Request(URL, headers={"Range": f"bytes={self.offset}-{last}"})
        with urlopen(req, timeout=45) as resp:
            assert resp.status == 206 and resp.headers["Content-Range"] == f"bytes {self.offset}-{last}/{SIZE}"
            data = resp.read(limit+1)
        assert len(data) == limit
        self.offset += limit
        return data


def discard_tensor(*_):
    return None


class ConfigReader(pickle.Unpickler):
    """Allow only primitive configuration data, skipping all tensor storages."""

    def persistent_load(self, identity):
        assert identity[0] == "storage"
        return None

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_tensor_v3"):
            return discard_tensor
        raise pickle.UnpicklingError(f"Unrecognized checkpoint metadata type {module}.{name}")


def run(destination):
    with zipfile.ZipFile(CheckpointIndex(), mode="r") as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        (metadata,) = names
        p = archive.read(metadata)
    layout = ConfigReader(io.BytesIO(p)).load()["model_config"]
    assert isinstance(layout, dict) and {"encoder", "neck", "depth_head", "mask_head"} <= layout.keys()
    checkpoint = dict(revision=REV, source=URL, published_checkpoint_bytes=SIZE, metadata_sha256=sha256(p).hexdigest(),
                      model_config=layout)
    destination.write_text(json.dumps(checkpoint, indent=2, allow_nan=False) + "\n")
    print("Published LingBot configuration extracted", sha256(p).hexdigest(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
