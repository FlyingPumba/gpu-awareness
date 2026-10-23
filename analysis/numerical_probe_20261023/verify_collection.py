"""Verify the saved probe artifacts against their remote originals and exits."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[2]
TASK=REPO.parents[1]
RAW=TASK/'outputs/numerical-probes/20261023'
SOURCE=Path(__file__).parent
remote='''import hashlib,json,subprocess,sys
from pathlib import Path
root=Path('/workspace/numerical-probe-20261023');role=sys.argv[1]
files=list((root/role).iterdir())
files += [p for p in root.iterdir() if p.is_file() and (p.suffix in ('.log','.exit','.pid','.py') or p.name in ('patch-'+role+'.json','dataset-audit.json'))]
items=[]
for p in sorted(files):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 items.append(dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(files=items,exits={n:(root/n).read_text().strip() for n in ('probe.exit','patch.exit','patch-retry.exit')},
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],text=True).strip(),
 gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip())))
'''
receipt={'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hosts':{}}
for host,role in [('runpod3','a100'),('runpod2','rtx')]:
    r=subprocess.run(['ssh',host,'/workspace/gpu-fingerprint/.venv/bin/python - '+role],input=remote,
                     text=True,capture_output=True,check=True)
    d=json.loads(r.stdout)
    assert d['exits']=={'probe.exit':'0','patch.exit':'1','patch-retry.exit':'0'}
    assert d['gpu_processes']==''
    for item in d['files']:
        name=Path(item['path'])
        if len(name.parts)>1:local=RAW/name
        elif name.suffix=='.py':local=SOURCE/name
        elif name.suffix in ('.log','.exit','.pid'):local=RAW/(role+'-process')/name
        else:local=RAW/name
        assert local.stat().st_size==item['size'],str(local)
        assert hashlib.sha256(local.read_bytes()).hexdigest()==item['sha256'],str(local)
    receipt['hosts'][host]=d
receipt['initial_patch_failure']='Donor file transfer had not completed; failure occurred before model load. Logs preserved; verified retries exited 0.'
receipt['raw_bytes_verified']=sum(x['size'] for d in receipt['hosts'].values() for x in d['files'])
receipt['files_verified']=sum(len(d['files']) for d in receipt['hosts'].values())
target=REPO/'results/numerical-probe-20261023/collection-verification.json'
target.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='hosts'},indent=2))
