import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

def configure():
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    root = ROOT / 'vendor/LIBERO/libero/libero'
    config = ROOT / 'configs/libero'
    config.mkdir(parents=True, exist_ok=True)
    paths = {'benchmark_root': root, 'bddl_files': root / 'bddl_files', 'init_states': root / 'init_files', 'assets': root / 'assets', 'datasets': ROOT / 'work/datasets'}
    # JSON string literals are valid YAML and safely quote Windows drive paths.
    import json
    (config / 'config.yaml').write_text('\n'.join(k + ': ' + json.dumps(str(v)) for k, v in paths.items()) + '\n', encoding='utf-8')
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'egl')
    os.environ['ROBOSUITE_LOG_PATH'] = str(ROOT/'work/robosuite.log')
    sys.path.insert(0, str(ROOT / 'vendor/LIBERO'))
    return paths
