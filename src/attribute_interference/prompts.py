"""Factorial prompts and directed minimal pairs. No model dependencies."""
import hashlib
import random
from pathlib import Path
from .storage import atomic_json, digest, read_csv, source_hash, verify_run, write_csv

FACTORS = ['C', 'N', 'P', 'M', 'O2']
OBJECTS = ['cup', 'bowl', 'vase', 'chair', 'bottle', 'spoon', 'clock', 'table',
           'box', 'lamp', 'ball', 'bucket', 'tray', 'stool', 'bench', 'toy car']
ANIMALS = ['dog', 'cat', 'bird', 'horse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'black']
MATERIALS = ['wooden', 'metal', 'glass', 'plastic']
SECOND_OBJECTS = ['plate', 'book', 'umbrella', 'backpack']
PLURALS = {'box': 'boxes', 'bench': 'benches'}
TEMPLATE_VERSION = '1'


def plural(obj):
    return PLURALS.get(obj, obj + 's')


def assignment(obj, seed):
    # Per-object RNG: assignment does not depend on subset/order of objects.
    rng = random.Random(f'{seed}:{obj}')
    color = rng.choice(COLORS)
    return {'color': color, 'count': rng.choice([2, 3, 4]),
            'position': rng.choice(['left', 'right']), 'material': rng.choice(MATERIALS),
            'second_object': rng.choice(SECOND_OBJECTS),
            'second_color': rng.choice([c for c in COLORS if c != color])}


def render(obj, active, values):
    noun = plural(obj) if 'N' in active else obj
    words = []
    if 'N' in active:
        words.append({2: 'two', 3: 'three', 4: 'four'}[values['count']])
    if 'C' in active:
        words.append(values['color'])
    if 'M' in active:
        words.append(values['material'])
    words.append(noun)
    if 'N' not in active:
        words.insert(0, 'an' if words[0][0] in 'aeiou' else 'a')
    text = 'a photo of ' + ' '.join(words)
    if 'P' in active:
        text += f" on the {values['position']} side of the image"
    if 'O2' in active:
        text += f" and a {values['second_color']} {values['second_object']}"
    return text


def build(cfg):
    split = cfg['dataset']['split']
    if split == 'main':
        objects, animals = OBJECTS, ANIMALS
    elif split == 'pilot':
        objects, animals = ['mug', 'basket'], ['rabbit']
    else:
        objects, animals = ['teapot'], []
    rows, pairs = [], []
    # Eight technical cases include singleton C/N/P/M/O2 and CN.
    smoke_masks = [0, 1, 2, 4, 8, 16, 3, 31]
    for obj in objects + animals:
        factors = [f for f in FACTORS if obj not in animals or f != 'M']
        values = assignment(obj, cfg['dataset']['assignment_seed'])
        if obj in animals:
            values['material'] = ''
        ids = {}
        masks = smoke_masks if split == 'smoke' else range(2 ** len(factors))
        for mask in masks:
            active = [f for bit, f in enumerate(factors) if mask & (1 << bit)]
            pid = digest({'object': obj, 'active': active, 'values': values,
                          'template': TEMPLATE_VERSION})[:16]
            ids[mask] = pid
            rows.append({'prompt_id': pid, 'split': split, 'object': obj,
                         'active_factors': '|'.join(active), 'complexity': len(active),
                         **values, 'template_version': TEMPLATE_VERSION,
                         'prompt': render(obj, active, values)})
        for mask, pid in ids.items():
            for bit, factor in enumerate(factors):
                other = mask | (1 << bit)
                if other != mask and other in ids:
                    pairs.append({'base_id': pid, 'added_id': ids[other], 'added_factor': factor})
    return rows, pairs


def load_prompts(root):
    rows = read_csv(Path(root) / 'prompts.csv')
    for row in rows:
        row['active'] = row['active_factors'].split('|') if row['active_factors'] else []
        row['count'] = int(row['count'])
        row['complexity'] = int(row['complexity'])
    return rows


def prepare(cfg, root):
    root = Path(root)
    if (root / 'run.json').exists():
        return verify_run(cfg, root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Output directory is nonempty without run.json; use a fresh directory.')
    root.mkdir(parents=True, exist_ok=True)
    rows, pairs = build(cfg)
    write_csv(root / 'prompts.csv', rows)
    write_csv(root / 'pairs.csv', pairs)
    jobs = [{'image_id': f"{r['prompt_id']}_s{s}", 'prompt_id': r['prompt_id'], 'seed': s}
            for r in rows for s in cfg['dataset']['seeds']]
    write_csv(root / 'jobs.csv', jobs)
    run = {'config': cfg, 'config_hash': digest(cfg), 'source_hash': source_hash(),
           'prompt_count': len(rows), 'job_count': len(jobs), 'pair_count': len(pairs),
           'data_hashes': {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                           for n in ['prompts.csv', 'pairs.csv', 'jobs.csv']}}
    atomic_json(root / 'run.json', run)
    print(f'Prepared {len(rows)} prompts, {len(jobs)} jobs, {len(pairs)} minimal pairs.')
    return run
