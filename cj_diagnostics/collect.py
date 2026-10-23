"""Collect a completed probability-only plan and verify every collected byte."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def collect(role,host,remote,output):
    source=f'''import hashlib,json
from pathlib import Path
root=Path({remote!r})
assert (root/{(role+'.exit')!r}).read_text().strip()=='0'
assert json.loads((root/{(role+'-status.json')!r}).read_text())['event']=='completed'
entries=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and 'wandb' not in p.relative_to(root).parts:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  entries.append(dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(entries))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -'],input=source,
                          text=True,capture_output=True,check=True)
    entries=json.loads(result.stdout)
    dest=output/role
    dest.mkdir(parents=True,exist_ok=True)
    listing=output/(role+'-files.txt')
    listing.write_text(''.join(e['path']+'\n' for e in entries))
    subprocess.run(['rsync','-az','--files-from='+str(listing),host+':'+remote+'/',str(dest)+'/'],check=True)
    for e in entries:
        p=dest/e['path']
        assert p.stat().st_size==e['size'] and sha(p)==e['sha256'],str(p)
    provenance=json.loads((dest/(role+'-provenance.json')).read_text())
    assert sha(dest/'source/train.py')==provenance['trainer_sha256']
    (output/(role+'-manifest.json')).write_text(json.dumps(entries,indent=2)+'\n')
    return dict(role=role,files=len(entries),bytes=sum(e['size'] for e in entries),verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--remote',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker-host',default='runpod3')
    p.add_argument('--coordinator-host',default='runpod2')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(collect,role,host,args.remote,args.output)
              for role,host in [('worker',args.worker_host),('coordinator',args.coordinator_host)]]
        roles=[job.result() for job in jobs]
    result=dict(roles=roles,files=sum(r['files'] for r in roles),bytes=sum(r['bytes'] for r in roles))
    (args.output/'collection-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
