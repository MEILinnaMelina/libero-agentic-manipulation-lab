import hashlib
import json
from pathlib import Path
import os

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonable(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False), encoding='utf-8')
    os.replace(str(tmp), str(path))

def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, default=jsonable, allow_nan=False) + '\n')
        f.flush()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def code_hash(root):
    paths = sorted(p for d in ['src', 'scripts', 'configs'] for p in (root/d).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.in', '.lock'} and p.name!='frozen.json' and '__pycache__' not in p.parts)
    return hashlib.sha256(''.join(p.relative_to(root).as_posix()+digest(p) for p in paths).encode()).hexdigest()
