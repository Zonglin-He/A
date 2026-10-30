"""Extract scalar reference metrics from previously verified official baseline logs.
These are author logs, not new local measurements; no data or model inference.
"""
from pathlib import Path
import json,re,hashlib
B=Path(__file__).resolve().parents[3];S=B/'crossplatform/round3/p0/official_archive_metadata/final_6384/eval';rows=[]
for platform in ['waymo','drone','quad']:
 logs=list((S/('Val_'+platform)).glob('*/log.txt'));assert len(logs)==1
 p=logs[0];s=p.read_text();records=[]
 for head,mode,threshold,rest in re.findall(r'(last_|proposal_) Box given span \((soft-token|contrastive)\) Acc(0\.25|0\.50): ([^\n]+)',s):
  records.append({'head':head,'mode':'bbs' if mode=='soft-token' else 'bbf','threshold':float(threshold),'topk':{k:float(v) for k,v in re.findall(r'Top-(\d+): ([0-9.]+)',rest)}})
 n=re.findall(r'length of testing dataset: (\d+)',s);miou=re.findall(r'mIoU: ([0-9.]+)',s)
 rows.append({'platform':platform,'source_log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':int(n[-1]),'reported_accuracy_rounded_3decimals':records,'reported_legacy_7head_miou':float(miou[-1]),'scope':'author archived log, not local measurement; batch48/oldhardware/software/workerRNG, dynamicpadding; mixedcheckpoint','not_expected_bitexact_with_B1':True})
out=B/'research/round5/AUTHOR_LOG_REFERENCE.json';out.write_text(json.dumps({'rows':rows,'source':'official verified baseline ZIP log files, acquisition record in round3 P0','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print(json.dumps({'platform_counts':{x['platform']:x['n'] for x in rows},'legacy_miou':{x['platform']:x['reported_legacy_7head_miou'] for x in rows}}))
