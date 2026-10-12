"""Atomic records, immutable run identity and model revision resolution."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def records(root, kind):
    return {p.stem: read_json(p) for p in sorted((Path(root) / kind).glob('*.json'))}


def put_record(root, kind, key, value):
    atomic_json(Path(root) / kind / f'{key}.json', value)


def export_manifest(root):
    path = Path(root) / 'generations.jsonl'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in records(root, 'generation_records').values():
            f.write(json.dumps(row, sort_keys=True) + '\n')
    tmp.replace(path)


def source_hash():
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['dataset']['split'] not in {'smoke', 'pilot', 'main'}:
        raise ValueError('Unknown dataset split')
    seeds = cfg['dataset']['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if cfg['dataset']['split'] == 'main' and seeds != [0, 1, 2, 3]:
        raise ValueError('Main experiments require all four seeds [0, 1, 2, 3]')
    if cfg['generation']['family'] not in {'sd15', 'sdxl'}:
        raise ValueError('Generator family must be sd15 or sdxl')
    return cfg


def verify_run(cfg, root):
    path = Path(root) / 'run.json'
    if not path.exists():
        raise ValueError('Run prepare first')
    run = read_json(path)
    if run['config_hash'] != digest(cfg) or run['source_hash'] != source_hash():
        raise ValueError('Config or source changed. Use a new output directory; do not mix runs.')
    for name, expected in run['data_hashes'].items():
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared data changed: {name}. Use a new run.')
    return run


def model_revision(root, role, checkpoint, revision):
    """Resolve once to a commit, then resume that exact commit without querying main."""
    path = Path(root) / 'models' / f'{role}.json'
    identity = {'checkpoint': checkpoint, 'requested_revision': revision}
    if path.exists():
        saved = read_json(path)
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError('Model identity changed')
        return saved['resolved_revision']
    from huggingface_hub import model_info
    sha = model_info(checkpoint, revision=revision).sha
    atomic_json(path, {**identity, 'resolved_revision': sha})
    return sha


def record_environment(root, role, torch):
    packages = ['torch', 'torchvision', 'diffusers', 'transformers', 'accelerate',
                'huggingface-hub', 'safetensors', 'Pillow', 'numpy']
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
           'packages': {p: importlib.metadata.version(p) for p in packages}}
    path = Path(root) / 'environment' / f'{role}.json'
    if path.exists():
        previous = read_json(path)
        if any(previous[k] != env[k] for k in ['python', 'packages']):
            raise ValueError('Software versions changed; use the original environment or a new run.')
    else:
        atomic_json(path, env)
    environment_id = digest(env)
    atomic_json(Path(root) / 'environment' / 'sessions' / f'{environment_id}.json', env)
    return environment_id


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for inference. Open notebooks/colab.ipynb with a GPU runtime.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch


def valid_image(path, size=None, expected_hash=None):
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if size and image.size != tuple(size):
                return False
        return not expected_hash or hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash
    except (OSError, ValueError, SyntaxError):
        return False
