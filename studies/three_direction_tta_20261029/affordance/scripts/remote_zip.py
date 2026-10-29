"""Read public HTTP zip via verified byte ranges, without downloading training assets."""
import io,requests,zipfile,pathlib,json,argparse,hashlib
class RemoteFile(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.pos=0;self.cache={}
  r=requests.get(url,headers={'Range':'bytes=0-0'},timeout=60);r.raise_for_status();assert r.status_code==206
  self.size=int(r.headers['Content-Range'].split('/')[-1])
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,size=-1):
  if size<0:size=self.size-self.pos
  if not size:return b''
  start=self.pos;end=min(self.size,start+size)-1;key=(start,end)
  if key not in self.cache:
   print('range',start,end,flush=True)
   r=requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=90);r.raise_for_status()
   if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise RuntimeError('Range mismatch')
   self.cache[key]=r.content
  self.pos=end+1;return self.cache[key]
def safe_write(root,name,content):
 p=(root/name).resolve();assert p.is_relative_to(root.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
def main():
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--limit',type=int,default=16);a=p.parse_args()
 z=zipfile.ZipFile(RemoteFile(a.url));names=z.namelist();a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'archive_manifest.json').write_text(json.dumps([{'name':i.filename,'size':i.file_size,'compressed':i.compress_size} for i in z.infolist()],indent=2))
 txt=[n for n in names if n.endswith('.txt')];print('txt',txt,flush=True)
 for n in txt:
  if any(t in n.lower() for t in ['test','val']):safe_write(a.output,n,z.read(n))
 print('manifest lists saved',flush=True)
if __name__=='__main__':main()
