"""Selectively read the original MIT-States ZIP with verified HTTP byte ranges."""
import argparse
import hashlib
import io
import json
import re
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path
import requests
from experiment import write_json

URL='http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip'
CACHE=Path('.cache/mit_states')


def byte_range(url,start,end):
    with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90),stream=True) as response:
        response.raise_for_status()
        if response.status_code!=206:raise ValueError('Server did not honor bounded range')
        found=re.match(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
        if not found or int(found[1])!=start or int(found[2])!=end:raise ValueError('Unexpected range response')
        value=response.content
        if len(value)!=end-start+1:raise ValueError('Truncated range')
        return value,dict(response.headers)


class RemoteZip(io.RawIOBase):
    def __init__(self,url=URL):
        self.url=url;self.position=0
        with requests.get(url,headers={'Range':'bytes=0-3'},timeout=30,stream=True) as r:
            r.raise_for_status();self.size=int(r.headers['Content-Range'].split('/')[-1]);self.headers=dict(r.headers)
        self.cache={}
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        if not size:return b''
        end=min(self.size-1,self.position+size-1);key=(self.position,end)
        if key not in self.cache:self.cache[key]=byte_range(self.url,*key)[0]
        value=self.cache[key];self.position+=len(value);return value


def catalog():
    CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/'catalog.json'
    if path.exists():return json.loads(path.read_text())
    source=RemoteZip();archive=zipfile.ZipFile(source);files=[]
    for entry in archive.infolist():
        if not entry.filename.lower().endswith(('.jpg','.jpeg','.png')) or '__MACOSX' in entry.filename:continue
        parts=Path(entry.filename).parts
        if 'images' not in parts:continue
        pair=parts[parts.index('images')+1]
        if ' ' not in pair:continue
        state,noun=pair.split(' ',1)
        files.append(dict(path=entry.filename,state=state,noun=noun,offset=entry.header_offset,
                          compressed_size=entry.compress_size,size=entry.file_size,crc32=entry.CRC,compression=entry.compress_type))
    result=dict(url=URL,archive_size=source.size,last_modified=source.headers.get('Last-Modified'),etag=source.headers.get('ETag'),
                directory_sha256=hashlib.sha256(b''.join(source.cache.values())).hexdigest(),files=files)
    write_json(path,result);return result


def download_member(entry):
    # Local header's name/extra length is read independently from the central directory.
    header,_=byte_range(URL,entry['offset'],entry['offset']+29)
    if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
    name_size,extra_size=struct.unpack_from('<HH',header,26)
    start=entry['offset']+30+name_size+extra_size
    compressed,_=byte_range(URL,start,start+entry['compressed_size']-1)
    if entry['compression']==zipfile.ZIP_DEFLATED:payload=zlib.decompress(compressed,-15)
    elif entry['compression']==zipfile.ZIP_STORED:payload=compressed
    else:raise ValueError('Unsupported ZIP compression')
    if len(payload)!=entry['size'] or zlib.crc32(payload)!=entry['crc32']:raise ValueError('ZIP CRC/size mismatch')
    return payload


if __name__=='__main__':
    data=catalog();counts=Counter((f['state'],f['noun']) for f in data['files'])
    print('Images',len(data['files']),'states',len({s for s,n in counts}),'objects',len({n for s,n in counts}))
    print('States:',sorted({s for s,n in counts}))
    pairs=[('closed','open'),('empty','filled'),('clean','dirty'),('raw','cooked'),('whole','sliced'),('uncut','cut'),('ripe','unripe'),('peeled','unpeeled'),('folded','unfolded'),('solid','melted'),('frozen','thawed'),('bent','straight'),('broken','unbroken'),('new','worn'),('wet','dry')]
    for a,b in pairs:
        common=sorted(n for s,n in counts if s==a and counts[a,n]>=10 and counts[b,n]>=10)
        print(a,'→',b,':',len(common),common)
