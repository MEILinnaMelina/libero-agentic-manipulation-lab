"""Copy only research source/assets; never copy credentials, caches or run outputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {'.py', '.xml', '.obj', '.stl', '.msh', '.mtl', '.png', '.jpg', '.jpeg', '.bddl', '.pruned_init', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.cfg', '.ini'}
SKIP = {'.git', '__pycache__', '.pytest_cache', 'runs', 'outputs', 'videos', 'wandb', 'datasets', '.venv', 'node_modules'}

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])

def main():
    for name in ['src/libero_eval', 'configs/libero', 'scripts', 'tests', 'docs/provenance', 'vendor', 'runs', 'reports', 'work', 'work/datasets']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ['LIBERO', 'RoboEval-main']:
        repo = ROOT.parent / name
        tracked = set(git(repo, 'ls-files', '-z').decode().split('\0'))
        entries = []
        candidates = (list((repo / 'libero').rglob('*')) if name == 'LIBERO' else
                      list((repo / 'roboeval').rglob('*')) + list((repo / 'examples').glob('*agentic_v2.py')) + list((repo / 'docs').glob('*agentic_v2*')))
        candidates += [p for p in repo.iterdir() if p.is_file() and (p.name.startswith(('LICENSE', 'COPYING', 'README', 'requirements', 'setup')))]
        for path in sorted(set(candidates)):
            rel = path.relative_to(repo)
            if not path.is_file() or any(p in SKIP for p in rel.parts):
                continue
            if path.suffix.lower() not in SUFFIXES and not path.name.startswith(('LICENSE', 'COPYING')):
                continue
            if any(s in path.name.lower() for s in ['secret', 'credential', '.env', 'token', 'auth.json']):
                continue
            data = path.read_bytes()
            # Source files with accidental inline secrets must not be copied or diffed.
            if path.suffix in {'.py', '.json', '.yaml', '.yml', '.toml'}:
                import re
                if re.search(rb'sk-[A-Za-z0-9_-]{24,}', data):
                    raise RuntimeError('Credential-like content: ' + str(rel))
            dest = ROOT / 'vendor' / name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            entries.append({'path': rel.as_posix(), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'tracked': rel.as_posix() in tracked})
        for label, args in [('worktree', ['diff', '--binary', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]:
            data = git(repo, *args)
            import re
            data = re.sub(rb'sk-[A-Za-z0-9_-]{24,}', b'[REDACTED]', data)
            (ROOT / 'docs/provenance' / (name + '.' + label + '.txt')).write_bytes(data)
        summary[name] = {'source': str(repo), 'commit': git(repo, 'rev-parse', 'HEAD').decode().strip(), 'files': entries}
    (ROOT / 'docs/provenance/sources.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: {'files': len(v['files']), 'bytes': sum(x['bytes'] for x in v['files']), 'commit': v['commit']} for k, v in summary.items()}, indent=2))

if __name__ == '__main__':
    main()
