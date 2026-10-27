"""Display-only PDF text edit; preserve all existing image and geometry operators.

No model, prediction, GT, video, or experimental renderer is imported. Existing
published PDFs are the sole figure inputs. Old sources and private figures stay
unchanged. Run only after the active matched timing GPU process has exited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ContentStream, TextStringObject

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'outputs/figure_query_only_v1'
SOURCE_COMMIT = 'c9e023426916fd5f41043745a97d2831f4467a9f'
FIGURES = [
    ('good', f'results/review_strengthening_v1/mechanism_v1/figures/case_{i}.pdf', 2600, 1690)
    for i in range(1, 4)
] + [
    ('failure', f'results/failure_visualization_v1/figures/failure_{i:02d}_{kind}.pdf', 3360, 2100)
    for i, kind in enumerate(['temporal', 'spatial', 'final_DP', 'pooling_ROI'], 1)
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_of(args):
    return ''.join(str(x) for x in args[0] if isinstance(x, str))


def images(page):
    result = {}
    objects = page['/Resources'].get('/XObject', {})
    for name, ref in objects.items():
        obj = ref.get_object()
        if obj.get('/Subtype') == '/Image':
            result[str(name)] = hashlib.sha256(obj._data).hexdigest()
    return result


def edit(kind, rel, width, height):
    out = DEST / kind / Path(rel).name
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), 'Do not overwrite an existing display attempt'
    source = DEST / 'original_pdfs' / kind / Path(rel).name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(subprocess.check_output(['git', 'show', SOURCE_COMMIT + ':' + rel], cwd=ROOT))
    reader = PdfReader(source)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    def extracted_query(p):
        header = []
        cutoff = float(p.mediabox.height) * (.94 if kind == 'good' else .91)
        def visit(s, cm, tm, font, size):
            if cm[5] + tm[5] > cutoff and 'Query:' in s:
                header.append(s.split('Query:', 1)[1].strip())
        p.extract_text(visitor_text=visit)
        assert len(header) == 1
        return ''.join(header[0].split())
    original_query = extracted_query(page)
    stream = ContentStream(page.get_contents(), reader)
    original = list(stream.operations)
    changed = []
    removed = []
    query_blocks = 0
    for i, (args, op) in enumerate(original):
        if op != b'TJ':
            continue
        text = text_of(args)
        if text.startswith('Case ') and ' | Query:' in text:
            assert kind == 'good'
            query = 'Query:' + text.split(' | Query:', 1)[1]
            stream.operations[i] = ([ArrayObject([TextStringObject(query)])], op)
            changed.append(i)
            removed.append(text.split(' | Query:', 1)[0])
            query_blocks += 1
        elif text.startswith(('Actual saved predictions |', 'Failure ', 'Ours minus ')):
            stream.operations[i] = ([ArrayObject([TextStringObject('')])], op)
            changed.append(i)
            removed.append(text)
        elif text.startswith('Query:') or text.startswith('Query'):
            query_blocks += 1
    assert query_blocks == 1
    assert len(changed) == (2 if kind == 'good' else 2)
    for i, op in enumerate(original):
        if i not in changed:
            assert stream.operations[i] == op
    before_images = images(page)
    page.replace_contents(stream)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'Query-only case figure', '/Subject': 'Display-only header simplification'})
    with out.open('wb') as handle:
        writer.write(handle)
    check = PdfReader(out).pages[0]
    new_text = check.extract_text()
    assert images(check) == before_images
    assert 'Case ' not in new_text and 'Failure ' not in new_text
    assert 'Ours minus ' not in new_text and 'Post-hoc illustrative' not in new_text
    assert 'Actual saved predictions |' not in new_text
    assert new_text.count('Query') == 1
    # PDF text extraction can insert spaces at kerning adjustments. Compare the
    # actual caption as normalized extracted text; all other PDF operators were
    # already checked above, and the entire body is also pixel-checked below.
    assert original_query == extracted_query(check)
    png = out.with_suffix('.png')
    subprocess.run(['pdftoppm', '-singlefile', '-png', '-scale-to-x', str(width),
                    '-scale-to-y', str(height), str(out), str(png.with_suffix(''))], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Same renderer and resolution: all pixels below the header must remain exact.
    original_preview = out.with_name(out.stem + '_original_check.png')
    subprocess.run(['pdftoppm', '-singlefile', '-png', '-scale-to-x', str(width),
                    '-scale-to-y', str(height), str(source), str(original_preview.with_suffix(''))],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    from PIL import Image, ImageChops
    with Image.open(original_preview) as a, Image.open(png) as b:
        y = round(height * (.075 if kind == 'good' else .10))
        assert a.size == b.size == (width, height)
        assert ImageChops.difference(a.crop((0, y, width, height)), b.crop((0, y, width, height))).getbbox() is None
    original_preview.unlink()
    return dict(source=rel, source_commit=SOURCE_COMMIT, source_sha256=sha(source), pdf=str(out.relative_to(ROOT)),
                PDF_sha256=sha(out), png=str(png.relative_to(ROOT)), PNG_sha256=sha(png),
                changed_text_operators=changed, untouched_other_PDF_operators=True,
                encoded_image_streams_exact=before_images, body_pixels_exact=True,
                header_only_query=True, original_query_preserved=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['render'])
    parser.parse_args()
    active = ROOT / 'outputs/matched_pipeline_timing_v1/runtime_v4/GPU_exit.json'
    assert active.exists(), 'Wait for the matched timing GPU to exit before PDF rendering'
    assert not (DEST / 'render_receipt.json').exists()
    rows = [edit(*item) for item in FIGURES]
    rec = dict(status='seven_query_only_figures_rendered_pending_visual_inspection', figures=rows,
               code_sha256=sha(__file__), new_model=0, new_Source=0, new_expert=0,
               new_optimizer=0, new_GT_score=0, unchanged_prediction_and_scientific_sources=True)
    (DEST / 'render_receipt.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(dict(status=rec['status'], figures=len(rows))))


if __name__ == '__main__':
    main()
