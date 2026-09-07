"""Documented platform-only corrections for the pinned robosuite 1.4.0 wheel."""
import sysconfig
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
site=Path(sysconfig.get_paths()['purelib'])
patches={
    'robosuite/utils/log_utils.py': [('logging.FileHandler("/tmp/robosuite.log")', 'logging.FileHandler(__import__("os").environ.get("ROBOSUITE_LOG_PATH", __import__("os").path.join(__import__("tempfile").gettempdir(), "robosuite.log")))')],
    'robosuite/utils/binding_utils.py': [('ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))', 'ctypes.WinDLL(os.path.join(os.path.dirname(mujoco.__file__), "mujoco.dll"))'), ('if macros.MUJOCO_GPU_RENDERING and os.environ.get', 'if _SYSTEM != "Windows" and macros.MUJOCO_GPU_RENDERING and os.environ.get')]
}
records=[]
for name,changes in patches.items():
    p=site/name
    original=p.read_text(encoding='utf-8')
    data=original
    for old,new in changes:
        if old not in data and new not in data:
            raise RuntimeError('Patch target missing: '+name)
        data=data.replace(old,new)
    backup=ROOT/'docs/provenance'/('upstream-'+p.name)
    if not backup.exists(): backup.write_text(original,encoding='utf-8')
    p.write_text(data,encoding='utf-8')
    records.append({'file':name,'original_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'patched_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':changes})
(ROOT/'docs/provenance/windows_patches.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(records)
