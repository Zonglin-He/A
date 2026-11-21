"""Read the visible outline twice without displaying endpoints or a signed axis."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from endpoints import HERE, OUT, PRIVATE, ancestry, digest, read, write
from landmarks import check_assets

PROMPT = ('Trace the VISIBLE outer silhouette of the central object referred to by this tight target crop. '
          'Follow its actual visible boundary in the photograph. Do not trace the crop rectangle, the background, '
          'or guess hidden parts. If the central target or its boundary cannot be identified, return '
          '{"polygon":null}. Otherwise return only {"polygon":[[x,y],...]} with 6 to 12 boundary vertices '
          'in perimeter order, normalized separately from 0 to 1000 (left to right and top to bottom). '
          'Do not repeat the first vertex. No explanation.')


def eligible(obs):
    if obs["status"] != "eligible":
        return False
    l,t,r,b = obs["views"][1]["rect"]
    return all(l <= x < r and t <= y < b for x,y in obs["correct"])


def crops(row, obs):
    assert digest(row["rgb"]) == row["rgb_sha"]
    with Image.open(row["rgb"]) as image:
        crop = image.convert("RGB").crop(tuple(obs["views"][1]["rect"]))
    return [crop, ImageOps.mirror(crop)]


@torch.inference_mode()
def run(model_dir):
    start = perf_counter()
    panel,_,_,_ = ancestry()
    input_seal = read(OUT/"input_seal.json")
    assert input_seal["observations"] == digest(PRIVATE/"observations.json")
    observed = read(PRIVATE/"observations.json")
    rows = [f for f in panel if eligible(observed[f["uid"]])]
    assert len(rows) == 239
    paired = [f["uid"] for f in rows]
    pins = dict(protocol=digest(HERE/"OUTLINE_PROTOCOL.json"), code=digest(__file__),
                original_inputs=digest(OUT/"input_seal.json"), prompt=hashlib.sha256(PROMPT.encode()).hexdigest(),
                primary_seal=digest(OUT/"prediction_seal.json"), revision=check_assets(model_dir))
    directory = PRIVATE/"outline_replies"
    directory.mkdir(exist_ok=True,mode=0o700)
    processor = AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    processor.image_processor.size = {"shortest_edge":65536,"longest_edge":524288}
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,use_safetensors=True,
        dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval().requires_grad_(False)
    prompt = processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}],
                                            tokenize=False,add_generation_prompt=True)
    fresh = 0
    for j in range(0,len(rows),6):
        missing = [f for f in rows[j:j+6] if not (directory/(f["uid"]+".json")).exists()]
        if missing:
            images = [im for f in missing for im in crops(f,observed[f["uid"]])]
            batch = processor(text=[prompt]*len(images),images=images,padding=True,return_tensors="pt").to("cuda")
            ids = model.generate(**batch,do_sample=False,max_new_tokens=236)
            answers = processor.batch_decode(ids[:,batch.input_ids.size(1):],skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            for i,f in enumerate(missing):
                write(directory/(f["uid"]+".json"),dict(uid=f["uid"],pins=pins,answers=answers[2*i:2*i+2],
                      images=[hashlib.sha256(im.tobytes()).hexdigest() for im in images[2*i:2*i+2]]))
            fresh += len(images)
        if j%30 == 0:
            print("Outline frames",min(j+6,len(rows)),"/",len(rows),flush=True)
    manifest = {}
    for uid in paired:
        path=directory/(uid+".json")
        payload=read(path)
        assert payload["pins"]==pins and payload["uid"]==uid and len(payload["answers"])==2
        manifest[uid]=digest(path)
    write(PRIVATE/"outline_hashes.json",manifest)
    write(OUT/"outline_raw_seal.json",dict(pins=pins,frames=len(rows),calls=2*len(rows),fresh_calls=fresh,
          replies=digest(PRIVATE/"outline_hashes.json"),seconds=perf_counter()-start,gpu=torch.cuda.get_device_name(),
          torch=torch.__version__,transformers=transformers.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          target_labels_read=False,FINAL_access=False))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir",type=Path)
    run(cli.parse_args().model_dir)
