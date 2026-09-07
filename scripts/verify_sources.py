import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.io import read,digest,dump
manifest=read(ROOT/'docs/provenance/sources.json')
errors=[]
for name,source in manifest.items():
    for entry in source['files']:
        p=ROOT/'vendor'/name/entry['path']
        if not p.exists() or digest(p)!=entry['sha256']: errors.append(str(p))
dump(ROOT/'reports/source_integrity.json',{'ok':not errors,'mismatches':errors,'files_checked':sum(len(s['files']) for s in manifest.values())})
print('source integrity:',not errors)
sys.exit(bool(errors))
