"""Single interface correction: enforce predeclared JSON grammar during generation.

The v1 unconstrained-format failures and all prior utility remain preserved.
No prompt, model, image, coordinate convention or gate changes.
"""
import collections,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from scripts.decota_corrective_opd_v1 import OUT as BASE,MODEL,verify,parse
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=BASE/'structured_v2'

def prefix_ok(s,feedback):
    start='{"action":'
    if len(s)<len(start):return start.startswith(s)
    if not s.startswith(start):return False
    suffix=s[len(start):]
    if not suffix:return True
    action=suffix[0]
    if action not in ('012' if feedback else '12'):return False
    if action in '02':
        expected=action+',"box":null}'
        return expected.startswith(suffix)
    intro='1,"box":['
    if len(suffix)<len(intro):return intro.startswith(suffix)
    if not suffix.startswith(intro):return False
    t=suffix[len(intro):];offset=0
    for coordinate in range(4):
        begin=offset
        while offset<len(t) and t[offset].isdigit():offset+=1
        digits=t[begin:offset]
        if not digits:return offset==len(t)
        if len(digits)>4 or len(digits)>1 and digits[0]=='0' or int(digits)>1000:return False
        if offset==len(t):return True
        if coordinate==3:return ']}'.startswith(t[offset:])
        if t[offset]!=',':return False
        offset+=1
    return False

def finished(s,feedback):
    if not prefix_ok(s,feedback):return False
    try:x=json.loads(s)
    except ValueError:return False
    return set(x)=={'action','box'}

class Grammar:
    def __init__(self,tokenizer,eos):
        self.tokenizer=tokenizer;self.eos=[eos] if isinstance(eos,int) else list(eos);self.cache={}
        alphabet=set('{"action":0123456789,boxnul[]}');self.pieces=[]
        for token in range(len(tokenizer)):
            value=tokenizer.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if value and set(value)<=alphabet and token not in tokenizer.all_special_ids:self.pieces.append((token,value))
        assert self.pieces
    def constraint(self,start,feedback):
        def allowed(batch,tokens):
            s=self.tokenizer.decode(tokens[start:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
            key=(s,feedback)
            if key not in self.cache:
                if finished(s,feedback):out=self.eos
                else:out=[i for i,v in self.pieces if prefix_ok(s+v,feedback)]
                assert out,('JSON grammar dead end',s)
                self.cache[key]=out
            return self.cache[key]
        return allowed

def prepare():
    verify();OUT.mkdir(exist_ok=True)
    for name in ['INPUTS.json','TEACHER_JOBS.json','CAPTURE_BARRIER.json','native','images','ENGINEERING_READBACK.json','PARSER_ENGINEERING.json']:
        target=OUT/name
        if not target.exists():target.symlink_to(BASE/name,target_is_directory=(BASE/name).is_dir())
    # State-machine tests cover multi-character tokens and boundary numbers.
    examples=['{"action":0,"box":null}','{"action":2,"box":null}','{"action":1,"box":[0,23,999,1000]}']
    for s in examples:
        assert finished(s,True)
        assert all(prefix_ok(s[:i],True) for i in range(len(s)+1))
        assert not prefix_ok(s+'0',True)
    for bad in ['{"action":3','{"action":1,"box":[00','{"action":1,"box":[1001','{"action":1,"box":[1.2','{"action":0,"box":null}']:
        assert not prefix_ok(bad,False)
    write(OUT/'GRAMMAR_ENGINEERING.json',dict(status='pass',no_geometry_filter=True,integer_range=[0,1000],literal_action_tokens=True,outputs=examples))
    write(BASE/'amendments/002_structured_generation_after_v1.json',dict(reason='v1 used prompt-only formatting, which did not implement the requested fixed structured generation. Add one deterministic JSON grammar; rerun every paired input once. Original predictions, failure report and utility preserved.',
          utility_already_exposed=True,prompt_images_model_gate_unchanged=True,max_new_tokens_unchanged=96,
          pins={str(Path(__file__).resolve()):sha(__file__)},canonical_teacher_output=str(OUT),v1_decision_sha=sha(BASE/'TEACHER_DECISION.json')))

def run():
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from scripts.run_final_simplification_v1 import lease
    p=verify();assert sha(BASE/'TEACHER_JOBS.json')==read(BASE/'CAPTURE_BARRIER.json')['jobs_sha']
    guard=lease();start=time.perf_counter();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();files=[];count=collections.Counter()
    try:
        model,info=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'];model.eval().requires_grad_(False);pr=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        grammar=Grammar(pr.tokenizer,model.generation_config.eos_token_id);versions=[a._version for a in model.parameters()]
        for j,job in enumerate(read(BASE/'TEACHER_JOBS.json')):
            for f,h in zip(job['images'],job['image_shas']):assert sha(f)==h
            ims=[Image.open(f).convert('RGB') for f in job['images']];content=[dict(type='image') for _ in ims]+[dict(type='text',text=job['prompt'])]
            text=pr.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            inputs=pr(text=[text],images=ims,do_resize=False,return_tensors='pt').to('cuda');t=time.perf_counter();n=inputs['input_ids'].shape[1];feedback=job['mode']=='feedback'
            with torch.inference_mode():result=model.generate(**inputs,max_new_tokens=p['teacher_max_tokens'],do_sample=False,use_cache=True,prefix_allowed_tokens_fn=grammar.constraint(n,feedback))
            tokens=result[0,n:].cpu().tolist();answer=pr.tokenizer.decode(tokens,skip_special_tokens=True);parsed=parse(answer,job['original_box'],feedback)
            assert finished(answer,feedback),('structure failure',answer)
            # Invalid ordered geometry still abstains; grammar does not force a legal box or revise it.
            assert parsed['valid'] or parsed['reason']=='invalid_geometry'
            f=OUT/'teacher_results'/(hashlib.sha256(job['id'].encode()).hexdigest()+'.json')
            write(f,dict(id=job['id'],text=answer,tokens=tokens,parsed=parsed,seconds=time.perf_counter()-t,prompt_tokens=n,generated_tokens=len(tokens),GT_used=False,grammar=True))
            files.append(dict(path=str(f),sha=sha(f)));count[job['mode']]+=1;count['generated_tokens']+=len(tokens)
            if j%16==0:print('STRUCTURED',j,parsed['action'],parsed['valid'],'seconds',round(time.perf_counter()-start,1),flush=True)
            past=sum(read(f)['seconds'] for f in (BASE/'worker_receipts').glob('*.json'))
            assert time.perf_counter()-start+past<p['max_seconds'] and torch.cuda.max_memory_allocated()<p['peak_bytes']
            del inputs,result;gc.collect()
        assert versions==[a._version for a in model.parameters()]
        write(OUT/'TEACHER_BARRIER.json',dict(files=files,counts=dict(count),GT_used=False,model_unchanged=True,worker_seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),grammar_code_sha=sha(__file__),v1_utility_was_exposed=True))
    finally:
        write(OUT/'worker_receipts'/f'teacher_{time.time_ns()}.json',dict(seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),counts=dict(count)));guard.close()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
