"""Prepare only the two needed vendor modules from the pinned upstream checkout."""
import pathlib,subprocess,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];upstream=ROOT/'upstream';commit='8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()==commit
checks={'GREAT.py':'c4e224aeb923b651e7355ee8c4fc41bae83e3122871ed61ff91c34b1cddbae15','pointnet2_utils.py':'65a0f468020c839dba5c2db30955b53d8dedf4215e707123e19b1b54ef17990c'}
dest=ROOT/'adapted/model';dest.mkdir(parents=True,exist_ok=True)
for name,digest in checks.items():
 raw=(upstream/'model'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
 if name=='GREAT.py':
  text=raw.decode();text=text.replace('from transformers import AutoModel, AutoTokenizer','from transformers import AutoModel, AutoTokenizer, AutoConfig\nimport os');text=text.replace("AutoModel.from_pretrained('/PATH/roberta-base')","AutoModel.from_config(AutoConfig.from_pretrained(os.environ['GREAT_TEXT_MODEL'], local_files_only=True))");text=text.replace("AutoTokenizer.from_pretrained('/PATH/roberta-base')","AutoTokenizer.from_pretrained(os.environ['GREAT_TEXT_MODEL'], local_files_only=True)");text=text.replace('tokenized_phrases.to(device)','tokenized_phrases.to(next(self.text_encoder.parameters()).device)').replace('tokenized_queries.to(device)','tokenized_queries.to(next(self.text_encoder.parameters()).device)');raw=text.encode()
 target=dest/name
 if target.exists() and target.read_bytes()!=raw:raise RuntimeError('Refusing to overwrite a different adapted source: '+str(target))
 target.write_bytes(raw)
(dest/'__init__.py').touch();print('Prepared pinned source with the reviewed compatibility patch.')
