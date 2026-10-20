"""Read public split ZIP/TAR archives without downloading unrelated video sets."""
import bisect
import io
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_assets import get_range


class SplitHTTPFile(io.RawIOBase):
    def __init__(self, parts):
        self.parts = parts
        self.offsets = [0]
        for _, size in parts:
            self.offsets.append(self.offsets[-1]+size)
        self.position = 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.offsets[-1] if whence == 2 else 0)
        if self.position < 0:
            raise ValueError('Negative archive offset')
        return self.position

    def read(self, size=-1):
        size = min(self.offsets[-1]-self.position, size if size >= 0 else self.offsets[-1])
        output = []
        while size > 0:
            i = bisect.bisect_right(self.offsets, self.position)-1
            local = self.position-self.offsets[i]
            count = min(size, self.parts[i][1]-local, 8 << 20)
            key = (i, local, count)
            if key not in self.cache:
                data, _ = get_range(self.parts[i][0], local, local+count-1, self.parts[i][1])
                if count < 2 << 20:
                    if len(self.cache) > 32:
                        self.cache.clear()
                    self.cache[key] = data
            else:
                data = self.cache[key]
            output.append(data)
            self.position += count
            size -= count
        return b''.join(output)


HC_PARTS = [(f'https://huggingface.co/datasets/CircleRadon/hc-stvg/resolve/main/video_parts_part_{suffix}', size)
            for suffix, size in [('aa', 42949672960), ('ab', 42949672960), ('ac', 29209288518)]]


def hc_zip():
    return zipfile.ZipFile(SplitHTTPFile(HC_PARTS))


def extract_member(z, name, dest):
    """ZIP verifies CRC while streaming; publish the file only after EOF/CRC."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix+'.part')
    with z.open(name) as source, temp.open('wb') as target:
        while True:
            block = source.read(8 << 20)
            if not block:
                break
            target.write(block)
    temp.replace(dest)
    return dest
