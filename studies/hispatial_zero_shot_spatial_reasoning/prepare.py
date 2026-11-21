"""Project the official inputs onto a separate, answer-free inference manifest."""
import argparse
import base64
import binascii
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from PIL import Image

from download import DATA, OUTPUT, PROTOCOL, read_json, sha256, write_json


def store_image(folder, data):
    key = hashlib.sha256(data).hexdigest()
    path = folder / key
    if not path.exists():
        path.write_bytes(data)
    return key


def tsv_rows():
    csv.field_size_limit(1_000_000_000)
    path = DATA / '3dsrbench' / '3dsrbench_v1_vlmevalkit_circular.tsv'
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        next(reader)
        yield from enumerate(reader)


def cv_rows(columns):
    import pyarrow.parquet as parquet

    fields = sorted({'task', *columns})
    for subset in ('3d', '2d'):
        path = DATA / 'cv_bench' / f'test_{subset}.parquet'
        index = 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=24, columns=fields):
            for row in batch.to_pylist():
                if subset == '3d' or row['task'] == 'Relation':
                    yield subset, index, row
                index += 1


def sr_prompt(row):
    opts = [f'{label}. {value.strip()}' for label, value in zip('ABCD', row[3:7]) if value.strip()]
    return row[2].strip() + ('\n' + '\n'.join(opts) if opts else '')


def emb_prompt(row):
    return row['question'] + ''.join(f'\n{letter}. {value}' for letter, value in zip('ABCDEF', row['answer_options']))


def sr_image(row, downloaded):
    data = row[9].strip()
    if len(data) > 10:
        try:
            binary = base64.b64decode(data, validate=False)
            with Image.open(BytesIO(binary)) as image:
                image.load()
            return binary, 'rgb'
        except (binascii.Error, OSError, ValueError):
            pass
    url = row[11].strip()
    domain = urlsplit(url)
    if not url or domain.scheme not in ('http', 'https'):
        return None, 'image_failure'
    # Only the dataset's published COCO image fallback is accepted.
    if domain.hostname not in ('images.cocodataset.org', 'cocodataset.org'):
        raise ValueError(f'Unexpected 3DSRBench image fallback host: {domain.hostname}')
    if url not in downloaded:
        safe_url = url.replace('http://', 'https://', 1) if url.startswith('http://') else url
        with urlopen(safe_url, timeout=40) as response:
            final_host = urlsplit(response.geturl()).hostname
            if final_host not in ('images.cocodataset.org', 'cocodataset.org'):
                raise ValueError(f'Unexpected image redirect host: {final_host}')
            downloaded[url] = response.read()
        with Image.open(BytesIO(downloaded[url])) as image:
            image.load()
    return downloaded[url], 'flip_rgb' if '-flip' in row[1].strip() else 'rgb'


def cases(benchmark):
    if benchmark == 'cv_bench':
        for subset, index, row in cv_rows(('image', 'prompt')):
            binary = row['image']['bytes']
            if not binary:
                raise ValueError(f'CV-Bench image bytes absent for {subset}/{index}')
            yield f'{subset}/{index}', binary, 'cv_exif_rgb', row['prompt']
    elif benchmark == '3dsrbench':
        downloaded = {}
        for index, row in tsv_rows():
            if len(row) < 12:
                print(f'official skip at TSV row {index} with {len(row)} columns', flush=True)
                continue
            image, transform = sr_image(row, downloaded)
            yield f'tsv/{index}', image, transform, sr_prompt(row)
    else:
        path = DATA / benchmark / 'embspatial_bench.json'
        for index, row in enumerate(read_json(path)):
            binary = base64.b64decode(row['image'])
            yield f'json/{index}', binary, 'rgb', emb_prompt(row)


def run(benchmark):
    directory = OUTPUT / benchmark
    folder = directory / 'images'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = DATA / benchmark / 'source.json'
    for asset in read_json(source)['files']:
        if sha256(source.parent / asset['file']) != asset['sha256']:
            raise ValueError(f'dataset asset hash mismatch: {asset["file"]}')
    rows = 0
    refs = set()
    manifest = directory / 'inputs.jsonl'
    partial = directory / 'inputs.jsonl.tmp'
    with partial.open('w', encoding='utf-8') as output:
        for sample_id, data, transform, prompt in cases(benchmark):
            img = store_image(folder, data) if data is not None else None
            refs.add((img, transform))
            record = {'id': sample_id, 'image_sha256': img, 'transform': transform, 'prompt': prompt}
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            rows += 1
    partial.replace(manifest)
    counts = {'benchmark': benchmark, 'rows': rows, 'distinct_image_inputs': len(refs - {(None, 'image_failure')}),
              'inputs_sha256': sha256(manifest), 'source_sha256': sha256(source),
              'protocol_sha256': sha256(PROTOCOL), 'input_fields': ['id', 'image_sha256', 'transform', 'prompt']}
    write_json(directory / 'inputs_seal.json', counts)
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=read_json(PROTOCOL)['order'])
    run(parser.parse_args().benchmark)
