"""Read checked members from the official ZIPs without downloading unused frames."""
from io import RawIOBase
import re
import struct
from time import sleep
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

import requests

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'


def byte_range(url, start, end):
    for attempt in range(5):
        try:
            with requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(10,50)) as response:
                extent = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if response.status_code == 206 and extent and tuple(map(int,extent.groups()[:2])) == (start,end):
                    data = response.content
                    if len(data) == end-start+1:
                        return data
        except requests.RequestException:
            pass
        sleep(.5*(attempt+1))
    raise IOError(f'Could not verify range {start}-{end} in {url}')


class RemoteZipFile(RawIOBase):
    def __init__(self, filename):
        self.url = BASE + filename
        with requests.get(self.url, headers={'Range':'bytes=0-0'}, timeout=20, stream=True) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise IOError(f'The archive does not accept ranges: {filename}')
            self.size = int(response.headers['Content-Range'].split('/')[-1])
        self.offset = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        result = offset + (0,self.offset,self.size)[whence]
        if result < 0:
            raise ValueError('Negative ZIP offset')
        self.offset = result
        return result

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.size,self.offset+count)
        if end <= self.offset:
            return b''
        data = byte_range(self.url,self.offset,end-1)
        self.offset=end
        return data


def index(filename):
    with RemoteZipFile(filename) as source, ZipFile(source) as archive:
        return source.url,{entry.filename:entry for entry in archive.infolist()}


def member(url, info):
    offset = info.header_offset
    length = 30 + len(info.filename.encode()) + 4096 + info.compress_size
    data = byte_range(url, offset, offset + length - 1)
    if data[:4] != b'PK\x03\x04':
        raise ValueError('Invalid local ZIP header')
    name_len, extra_len = struct.unpack_from('<HH',data,26)
    start = 30 + name_len + extra_len
    value = data[start:start+info.compress_size]
    if info.compress_type == ZIP_DEFLATED:
        value = zlib.decompress(value,-15)
    elif info.compress_type != ZIP_STORED:
        raise ValueError(f'Unsupported ZIP compression for {info.filename}')
    if len(value) != info.file_size or zlib.crc32(value) != info.CRC:
        raise ValueError(f'ZIP member checksum mismatch for {info.filename}')
    return value


def save_member(url, info, destination):
    if not destination.is_file():
        staged = destination.with_suffix('.partial')
        staged.write_bytes(member(url,info))
        staged.replace(destination)
    if destination.stat().st_size != info.file_size or zlib.crc32(destination.read_bytes()) != info.CRC:
        raise ValueError(f'Cached member checksum mismatch for {destination}')
