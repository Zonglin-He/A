"""One bounded frozen sparse-candidate verifier. No tracker or GT reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'artifacts/stvg_sparse_stvg_r1_semantic_opd_v1'
OLD = ROOT / 'artifacts/stvg_target_teacher_distillation_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, mutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not mutable: raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def remaining():
    return read(BASE / 'BUDGET_LOCK.json')['deadline_unix'] - time.time()


def check():
    if remaining() <= 0: raise TimeoutError('absolute new-scope three-hour budget exhausted')
    if shutil.disk_usage(ROOT).free < 30 * 1024**3: raise RuntimeError('disk floor reached')


def inference_guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, Path)):
        name = os.fsdecode(args[0]).lower()
        # Transformers builds lazy import maps by reading every model's Python
        # source. Reading that library code is not invoking a tracker/model.
        if '/site-packages/transformers/' in name and name.endswith('.py'): return
        if any(s in name for s in ('/annos/', '/annotations/', '/anno_v2/', 'private_case_bindings',
              'gt_label', 'gt_subset', '/rows.json', '/summary.json', 'test_annotations.json',
              'valv2_proc.json', 'val_v2.json', 'train_v2.json', '/tracking/', '/checkpoints/stvg-r1-tracking/', '/site-packages/sam2/', '/external/grounded-sam-2/',
              'grounding_dino', 'grounding-dino')):
            raise PermissionError('sparse inference forbids GT and detector/tracker assets')


def source_case(rec, enrich=True):
    import numpy as np
    import torch
    from PIL import Image
    path = ROOT / rec['source_case_path']
    if sha(path) != rec['source_case_sha256']: raise RuntimeError('cold source SHA changed')
    case = torch.load(path, map_location='cpu', weights_only=False)
    if case['GT_read'] or not case['original_source_only'] or case['DINO_read_or_call']:
        raise RuntimeError('invalid cold student custody')
    positions = np.rint(np.linspace(0, len(case['frame_ids']) - 1, 8)).astype(int).tolist()
    if len(set(positions)) != 8: raise RuntimeError('eight distinct student input frames unavailable')
    case['sparse_positions'] = positions
    if not enrich: return case
    request = read(OLD / 'canonical_inputs' / rec['case_id'] / 'REQUEST.json')
    if request['source_video_sha256'] != case['input']['video_sha256']:
        raise RuntimeError('canonical video is not the same source')
    video = Path(request['video_path'])
    if sha(video) != request['canonical_video_sha256']: raise RuntimeError('canonical video SHA changed')
    frame_map = {r['physical_frame_id']: r['local_frame_index'] for r in request['frame_map']}
    wanted = {frame_map[case['frame_ids'][p]]: p for p in positions}
    folder = BASE / 'sparse_inputs' / rec['case_id']; folder.mkdir(parents=True, exist_ok=True)
    # Sequential FFV1 decoding preserves the physical frame map and holds one frame.
    import cv2
    cap = cv2.VideoCapture(str(video)); local = 0; actual = []
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok: raise RuntimeError('canonical sparse physical frame unavailable')
            if local in wanted:
                p = wanted.pop(local); physical = case['frame_ids'][p]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                previous = case['source_RGB'].get(physical)
                if previous and previous['RGB_sha256'] != pixel_sha:
                    raise RuntimeError('canonical vs saved source pixel mismatch')
                dst = folder / f'{physical:06d}.png'
                if not dst.exists(): Image.fromarray(rgb).save(dst)
                with Image.open(dst) as image:
                    if hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() != pixel_sha:
                        raise RuntimeError('sparse PNG roundtrip mismatch')
                case['source_RGB'][physical] = {'path': str(dst.relative_to(ROOT)),
                    'sha256': sha(dst), 'RGB_sha256': pixel_sha}
                actual.append({'physical_frame_id': physical, 'position': p, 'RGB_sha256': pixel_sha,
                               'timestamp_s': physical / case['input']['fps']})
            local += 1
    finally: cap.release()
    write(folder / 'RECEIPT.json', {'status': 'pass', 'scope': 'same_cold_source_sparse_physical_pixels',
          'records': sorted(actual, key=lambda x: x['position']), 'canonical_sha256': sha(video),
          'count': 8, 'GT_read': False, 'CPU_only': True, 'time': time.time()}, mutable=True)
    return case


def prepare():
    import torch
    torch.set_num_threads(2)
    for rec in read(BASE / 'ROSTER_LOCK.json')['records']:
        if rec['P0']:
            check(); source_case(rec)
    write(BASE / 'SOURCE_STATE_CUSTODY_READBACK.json', {'status': 'pass', 'scope': 'P0_8_cold_sources_and_64_sparse_frames',
          'count': 8, 'GT_read': False, 'CPU_only': True, 'time': time.time()}, mutable=True)


def worker(stage):
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
    from scripts.stvg_sparse_r1_prompt_v1 import build_request, parse_output
    torch.set_num_threads(2)
    check(); runtime = read(BASE / 'RUNTIME.json')
    for name, expected in runtime['pins'].items():
        if sha(ROOT / name) != expected: raise RuntimeError('runtime source pin changed: ' + name)
    for name, expected in runtime['protected'].items():
        if sha(ROOT / name) != expected: raise RuntimeError('protected source bytes changed: ' + name)
    checkpoint = ROOT / 'checkpoints/STVG-R1-7B'
    receipt = read(checkpoint / 'OFFICIAL_RECEIPT.json')
    if receipt['status'] != 'pass': raise RuntimeError('official model not accepted')
    write(BASE / 'GPU_STATUS.json', {'status': 'loading_frozen_sparse_verifier', 'stage': stage,
          'pid': os.getpid(), 'done': 0, 'total': 48 if stage == 'P0' else 48, 'GT_read': False,
          'time': time.time()}, mutable=True)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                    min_pixels=3136, max_pixels=448*448)
    for component in (processor.image_processor, processor.video_processor):
        component.min_pixels=3136; component.max_pixels=448*448
        component.size=dict(shortest_edge=3136,longest_edge=448*448)
    processor.video_processor.do_sample_frames=False
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
              trust_remote_code=False, torch_dtype=torch.bfloat16, use_safetensors=True,
              low_cpu_mem_usage=True, device_map='auto', max_memory={0:'20GiB','cpu':'4GiB'},
              offload_folder=BASE/'offload', offload_state_dict=True, offload_buffers=True,
              attn_implementation='sdpa').eval().requires_grad_(False)
    if getattr(model, 'is_quantized', False): raise RuntimeError('unquantized model required')
    if sum(p.numel() for p in model.parameters()) != 8292166656:
        raise RuntimeError('official complete model parameter count differs')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('teacher must remain frozen and in eval mode')
    device = model.get_input_embeddings().weight.device
    write(BASE / 'MODEL_LOAD_RECEIPT.json', {'status':'loaded','pid':os.getpid(),'GT_read':False,
          'model_load_seconds':time.perf_counter()-started,'device_map':{k:str(v) for k,v in model.hf_device_map.items()},
          'dtype':'bfloat16','backend':'sdpa','time':time.time()}, mutable=True)
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs): return remaining() <= 0
    all_rows = read(BASE/'ROSTER_LOCK.json')['records']
    rows = [r for r in all_rows if r['P0']] if stage=='P0' else all_rows
    outputs=[]; done=0
    for rec in rows:
        check(); case=source_case(rec)
        peers=[r for r in rows if r['dataset']==rec['dataset']]
        other=peers[(peers.index(rec)+1)%len(peers)]
        wrong=source_case(other, enrich=False)['caption']
        for branch in ('temporal','spatial'):
            for control in (('base','id_swap','wrong_query') if stage=='P0' else ('base',)):
                check(); dst=BASE/stage/'outputs'/f"{rec['case_id']}_{branch}_{control}.json"
                if dst.exists():
                    result=read(dst); outputs.append({'path':str(dst.relative_to(ROOT)), 'sha256':sha(dst)})
                    done+=1; continue
                prior=BASE/'P0'/'outputs'/f"{rec['case_id']}_{branch}_base.json"
                if stage=='P1' and prior.exists():
                    accepted=read(prior)
                    if accepted['runtime_sha256']!=sha(BASE/'RUNTIME.json') or accepted['source_case_sha256']!=rec['source_case_sha256']:
                        raise RuntimeError('prior exact P0 base response binding differs')
                    result={**accepted,'exact_P0_base_reuse':True,'reused_response_path':str(prior.relative_to(ROOT)),
                            'reused_response_sha256':sha(prior),'new_teacher_calls':0}
                    write(dst,result);outputs.append({'path':str(dst.relative_to(ROOT)),'sha256':sha(dst)});done+=1
                    continue
                request=build_request(case,branch, id_permutation='reverse' if control=='id_swap' else None,
                                      query_override=wrong if control=='wrong_query' else None)
                messages=request['messages']; video=np.asarray(request['video_frames'])
                if video.ndim!=4 or video.shape[0]!=8 or video.shape[-1]!=3:
                    raise RuntimeError('expected eight uint8 THWC marked RGB frames')
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                t0=time.perf_counter()
                inputs=processor(text=[text],videos=[video.transpose(0,3,1,2)],fps=float(request['fps']),
                                 padding=False,return_tensors='pt',add_special_tokens=False)
                tokens=int(inputs['input_ids'].shape[1])
                if tokens>4096: raise RuntimeError('sparse visual token cap exceeded before GPU generation')
                inputs=inputs.to(device); torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                prep=time.perf_counter()-t0; t1=time.perf_counter()
                with torch.inference_mode():
                    generated=model.generate(**inputs,max_new_tokens=1024,do_sample=False,use_cache=True,
                                             stopping_criteria=StoppingCriteriaList([Deadline()]))
                torch.cuda.synchronize(); delay=time.perf_counter()-t1
                completion=generated[0,tokens:]; raw=processor.tokenizer.decode(completion,skip_special_tokens=True)
                try:
                    parsed={**parse_output(raw,request['metadata']['catalog']),'status':'valid'}
                except (ValueError,TypeError,KeyError) as exc:
                    # A malformed teacher answer is measurement evidence, not
                    # an excuse to omit this query or rewrite its prompt.
                    parsed={'status':'invalid_output','error_type':type(exc).__name__,'error':str(exc)}
                result={'status':'accepted_actual_sparse_response','case_id':rec['case_id'],'dataset':rec['dataset'],
                        'branch':branch,'control':control,'raw_output':raw,'parsed':parsed,
                        'request_metadata':request['metadata'],'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'input_token_count':tokens,'output_token_count':int(completion.numel()),
                        'output_truncated_at_cap':int(completion.numel())>=1024,'preprocess_seconds':prep,
                        'generate_seconds':delay,'total_seconds':prep+delay,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'source_case_sha256':rec['source_case_sha256'],'runtime_sha256':sha(BASE/'RUNTIME.json'),
                        'GT_read':False,'teacher_frozen':True,'teacher_optimizer_steps':0,'time':time.time()}
                write(dst,result); outputs.append({'path':str(dst.relative_to(ROOT)),'sha256':sha(dst)});done+=1
                write(BASE/'GPU_STATUS.json',{'status':'running','stage':stage,'pid':os.getpid(),'done':done,
                      'total':len(rows)*(6 if stage=='P0' else 2),'GT_read':False,'time':time.time()},mutable=True)
                del inputs,generated,completion; torch.cuda.empty_cache()
    write(BASE/f'{stage}_GLOBAL_READOUT_BARRIER.json',{'status':'complete','scope':stage+'_all_sparse_responses_sealed',
          'files':outputs,'count':len(outputs),'roster_sha256':sha(BASE/'ROSTER_LOCK.json'),
          'runtime_sha256':sha(BASE/'RUNTIME.json'),'GT_read':False,'time':time.time()})
    write(BASE/'STATUS.json',{'status':'pending_actual_root_protocol_and_postseal_quality_readback','stage':stage,
          'done':done,'total':done,'GT_read':False,'time':time.time()},mutable=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','P0','P1']);args=parser.parse_args()
    sys.addaudithook(inference_guard)
    try:
        if args.phase=='prepare': prepare()
        else: worker(args.phase)
    except BaseException as exc:
        write(BASE/'FAILURE_CAPTURE.json',{'status':'actual_worker_failed','stage':args.phase,'pid':os.getpid(),
              'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
              'GT_read':False,'time':time.time()},mutable=True)
        write(BASE/'STATUS.json',{'status':'failed_requires_root_engineering_review','stage':args.phase,
              'pid':os.getpid(),'GT_read':False,'time':time.time()},mutable=True)
        raise


if __name__=='__main__': main()
