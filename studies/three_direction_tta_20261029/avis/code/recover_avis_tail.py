"""Recover dictionary-independent tar members from a verified gzip suffix.

For non-byte-aligned DEFLATE starts, prepend *empty* blocks with the same bit
alignment. They emit no literals, so they do not falsely replace unknown history
with known bytes. Decode twice with all00/allFF initial 32KiB dictionaries.
Every equal output byte is independent of the unknown initial history because
DEFLATE output consists exclusively of literal bytes and LZ77 copies.
"""
import hashlib
import json
from pathlib import Path
import zlib
from functools import lru_cache
import os

ROOT=Path(__file__).resolve().parents[1] / 'recovery_workspace'


def bits(value,n):return [(value>>i)&1 for i in range(n)]
def pack(values):
    out=bytearray((len(values)+7)//8)
    for i,v in enumerate(values):out[i//8]|=v<<(i%8)
    return bytes(out)


def empty_dynamic():
    order=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
    b=[0]+bits(2,2)+bits(0,5)+bits(0,5)+bits(15,4)
    for symbol in order:b+=bits(1 if symbol in [0,1] else 0,3)
    b += [0]*256+[1,0]  # EOB length1; no distance codes.
    b += [0]           # The sole literal/length code is EOB.
    assert len(b)==333
    check=b.copy();check[0]=1
    assert zlib.decompress(pack(check),wbits=-15)==b''
    return b


@lru_cache(maxsize=8)
def prefix(alignment):
    if alignment%2:
        b=empty_dynamic();m=next(n for n in range(4) if (len(b)+10*n)%8==alignment)
    else:
        b=[];m=next(n for n in range(4) if 10*n%8==alignment)
    b += ([0,1,0]+[0]*7)*m
    assert len(b)%8==alignment
    return pack(b),len(b)


def aligned_fragment(data,offset,alignment,limit=None):
    head,n=prefix(alignment)
    tail=data[offset:None if limit is None else offset+limit]
    if not n:return tail
    if alignment==0:return head+tail
    return head[:-1]+bytes([(head[-1]&((1<<alignment)-1)) | (tail[0]&~((1<<alignment)-1))])+tail[1:]


def decode(fragment,initial,max_output):
    d=zlib.decompressobj(wbits=-15,zdict=bytes([initial])*32768)
    out=d.decompress(fragment,max_output)
    return out,d


def tar_members(a,b):
    needle=b'datasets/test.json\0'
    start=0
    while True:
        at=a.find(needle,start)
        if at<0:return
        start=at+1
        header=a[at:at+512]
        if len(header)!=512 or header!=b[at:at+512]:continue
        try:
            expected=int(header[148:156].strip(b'\0 '),8)
            length=int(header[124:136].strip(b'\0 '),8)
        except ValueError:continue
        if sum(header[:148])+8*32+sum(header[156:])!=expected:continue
        if header[156:157] not in [b'0',b'\0']:continue
        end=at+512+length
        payload=a[at+512:end]
        if len(payload)!=length or payload!=b[at+512:end]:continue
        obj=json.loads(payload)
        if len(obj.get('videos',[]))!=205 or len(obj.get('categories',[]))!=26:continue
        if not obj.get('annotations'):continue
        yield at,payload,obj


def main():
    mib=int(os.environ.get('AVIS_TAIL_MIB','8'));assert mib in [8,16]
    source=json.loads((ROOT/('suffix_source_16.json' if mib==16 else 'suffix_source.json')).read_text())
    data=(ROOT/f'tail_{mib}MiB.bin').read_bytes()
    assert hashlib.sha256(data).hexdigest()==source['sha256']
    empty_dynamic()  # Verify that every alignment prefix is valid and emits nothing.
    for j in range(8):prefix(j)
    # Find an actual block in the first128KiB; verify the *entire remaining stream*.
    attempted=0
    recovered_hash=None
    witnesses=[]
    skip_to=0
    for offset in range(min(128<<10,len(data)-2)):
        if offset<skip_to:continue
        word=data[offset] | (data[offset+1]<<8)
        for alignment in range(8):
            if ((word>>alignment)&7) not in [0,2,4]:continue
            attempted+=1
            try:
                short=aligned_fragment(data,offset,alignment,32768)
                preview,decoder=decode(short,0,1<<20)
                if len(preview)<8192 or decoder.eof:continue
                fragment=aligned_fragment(data,offset,alignment)
                a,da=decode(fragment,0,512<<20)
                # A real end must land exactly before the original gzip trailer.
                if not da.eof or da.unused_data!=data[-8:]:continue
                b,db=decode(fragment,255,512<<20)
                if not db.eof or db.unused_data!=data[-8:] or len(a)!=len(b):continue
                print('valid tail stream',offset,alignment,'output',len(a),flush=True)
                for at,payload,obj in tar_members(a,b):
                    payload_hash=hashlib.sha256(payload).hexdigest()
                    if recovered_hash is not None and payload_hash!=recovered_hash:
                        raise RuntimeError('Different valid block starts recovered different target payloads')
                    recovered_hash=payload_hash
                    witnesses.append({'compressed_bit_offset':(source['offset']+offset)*8+alignment,
                                      'decoded_bytes':len(a),'member_offset_in_decoded_tail':at})
                    path=ROOT/'test.json.recovered'
                    path.write_bytes(payload)
                    receipt={'status':'dictionary_independent_member_recovered',
                        'source_suffix_sha256':source['sha256'],
                        'archive_compressed_bit_offset':(source['offset']+offset)*8+alignment,
                        'empty_prefix_bits':prefix(alignment)[1],
                        'trailer_exact':True,'tar_header_checksum':True,
                        'payload_all00_allFF_identical':True,'payload_sha256':payload_hash,
                        'payload_bytes':len(payload),'videos':len(obj['videos']),
                        'categories':len(obj['categories']),'annotations':len(obj['annotations']),
                        'independent_block_start_witnesses':witnesses,
                        'full_archive_crosscheck':'pending; do not claim full archive download completed'}
                    if len(witnesses)>=2:
                        (ROOT/'RECOVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
                        print(json.dumps(receipt,indent=2));return
                    print('first complete member witness',payload_hash,flush=True)
                    skip_to=offset+16384
                    break
                if recovered_hash is not None:continue
                raise RuntimeError('Found a valid stream but no complete dictionary-independent test.json in suffix')
            except zlib.error:pass
        if offset and offset%16384==0:print('scanned',offset,'attempts',attempted,flush=True)
    raise RuntimeError('No eligible block/member recovered; whole archive acquisition remains authoritative')


if __name__=='__main__':main()
