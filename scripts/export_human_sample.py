"""Select the remaining human-labeled images and write blind review sheets."""
import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

TEST_COUNT = 150
SALT = 'human-sample-v1'
LABELS = ['object', 'C', 'N', 'P', 'M', 'O2']
FIELDS = ['image_id', 'prompt_id', 'prompt', 'active_factors', 'reviewer', *[f'human_{x}' for x in LABELS], 'notes']
RULES = ('Fill annotations.csv with true, false or unknown. Leave inactive factors blank. '
         'Color or material is true only if every copy of the main object has it. '
         'Count must match exactly. Position is true only if every copy is in the requested half. '
         'If the main object is absent, mark all active factors false. Automatic labels are hidden.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_run(root, split):
    root = Path(root)
    run = json.loads((root / 'run.json').read_text())
    if run['config']['dataset']['split'] != split:
        raise ValueError(f'{root}: expected a {split} run')
    for name, expected in run['data_hashes'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'{root}: {name} does not match run.json')
    prompts = {p['prompt_id']: p for p in read_csv(root / 'prompts.csv')}
    for p in prompts.values():
        p['active'] = p['active_factors'].split('|') if p['active_factors'] else []
    with open(root / 'generations.jsonl', encoding='utf-8') as f:
        generated = [json.loads(line) for line in f if line.strip()]
    return prompts, [g for g in generated if g['status'] == 'ok']


def select(candidates, prompts, count):
    """Greedy balance: complexity, factors within complexity, prompt, object, seed; salted hash breaks ties."""
    levels, factors, used_prompts, objects, seeds = Counter(), Counter(), Counter(), Counter(), Counter()

    def add(g):
        p = prompts[g['prompt_id']]
        c = len(p['active'])
        levels[c] += 1
        factors.update((c, f) for f in p['active'])
        used_prompts[g['prompt_id']] += 1
        objects[p['object']] += 1
        seeds[g['seed']] += 1

    pool = [g for g in candidates if prompts[g['prompt_id']]['active']]
    order = {g['image_id']: hashlib.sha256(f"{SALT}:{g['image_id']}".encode()).hexdigest() for g in pool}

    def score(g):
        p = prompts[g['prompt_id']]
        c = len(p['active'])
        return (levels[c], sum(factors[(c, f)] for f in p['active']), used_prompts[g['prompt_id']],
                objects[p['object']], seeds[g['seed']], order[g['image_id']])

    if len(pool) < count:
        raise ValueError(f'Need {count} images, only {len(pool)} available')
    selected = []
    for _ in range(count):
        best = min(pool, key=score)
        pool.remove(best)
        add(best)
        selected.append(best)
    return sorted(selected, key=lambda g: order[g['image_id']])


def export(root, folder, selected, prompts, title):
    root = Path(root)
    out = root / folder
    if out.exists():
        raise ValueError(f'{out} already exists; preserving human edits.')
    for g in selected:
        if sha256(root / g['path']) != g['sha256']:
            raise ValueError(f"{g['image_id']}: image does not match its generation record")
    cards = []
    rows = []
    for g in selected:
        p = prompts[g['prompt_id']]
        rows.append({'image_id': g['image_id'], 'prompt_id': p['prompt_id'], 'prompt': p['prompt'],
                     'active_factors': p['active_factors']})
        cards.append(f'<article><h2>{html.escape(g["image_id"])}</h2><p>{html.escape(p["prompt"])}</p>'
                     f'<img width="384" loading="lazy" src="../{html.escape(g["path"])}"></article>')
    out.mkdir()
    with open(out / 'annotations.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, FIELDS, restval='')
        writer.writeheader()
        writer.writerows(rows)
    (out / 'selection.json').write_text(json.dumps({g['image_id']: g['sha256'] for g in selected}, indent=2))
    (out / 'gallery.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
                                      f'<h1>{title}</h1><p>{RULES}</p>' + ''.join(cards), encoding='utf-8')
    return out


def summary(selected, prompts):
    active = [prompts[g['prompt_id']]['active'] for g in selected]
    return {'images': len(selected),
            'complexity': dict(sorted(Counter(len(a) for a in active).items())),
            'factors': dict(sorted(Counter(f for a in active for f in a).items())),
            'objects': len({prompts[g['prompt_id']]['object'] for g in selected}),
            'seeds': dict(sorted(Counter(g['seed'] for g in selected).items()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', required=True, help='main_sdxl run directory')
    args = parser.parse_args(argv)

    prompts, generated = load_run(args.main, 'main')
    test = select(generated, prompts, TEST_COUNT)
    print(export(args.main, 'review_test', test, prompts, 'Evaluator test review'), summary(test, prompts))


if __name__ == '__main__':
    main()
