import sys,pathlib,zipfile,json,collections
from remote_zip import RemoteFile
root=pathlib.Path(__file__).resolve().parents[1]
u='https://drive.usercontent.google.com/download?id=1yOV4KdRVWXRjXlXbYZ4_H9peoZIIKc0Y&export=download&confirm=t'
z=zipfile.ZipFile(RemoteFile(u));info={i.filename:{'offset':i.header_offset,'size':i.file_size,'compressed':i.compress_size,'method':i.compress_type,'crc':i.CRC} for i in z.infolist()};(root/'checkpoint_zip_index.json').write_text(json.dumps(info,indent=2))
groups=collections.defaultdict(list)
for n,i in info.items():
 if '/data/' in n:groups[(i['size'],i['crc'])].append(n)
large=[{'size':key[0],'crc':key[1],'names':names} for key,names in groups.items() if len(names)>1 and key[0]>1024*1024]
(root/'checkpoint_zip_duplicate_groups.json').write_text(json.dumps(large,indent=2));print('duplicate potential saved bytes',sum(r['size']*(len(r['names'])-1) for r in large),flush=True);print('large groups',len(large),flush=True)
