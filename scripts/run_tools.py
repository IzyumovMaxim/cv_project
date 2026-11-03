"""Shared loading and comparison helpers for the analysis scripts."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from attribute_interference.analysis import Bootstrap, ci, compute_metrics, holm, mcnemar_p
from attribute_interference.evaluation import correct_position, filter_boxes

FACTORS = ['C', 'N', 'P', 'M', 'O2']
DRAWS = 2000


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    return {'True': True, 'False': False}.get(value)


def load_prompts(root):
    prompts = {}
    for p in read_csv(Path(root) / 'prompts.csv'):
        p['active'] = p['active_factors'].split('|') if p['active_factors'] else []
        p['count'] = int(p['count'])
        p['complexity'] = int(p['complexity'])
        prompts[p['prompt_id']] = p
    return prompts


def load_labels(root):
    labels = {}
    for r in read_csv(Path(root) / 'labels.csv'):
        row = {'image_id': r['image_id'], 'prompt_id': r['prompt_id'], 'seed': int(r['seed']),
               'object': boolean(r['object']), 'owl_count': int(r['owl_count']) if r['owl_count'] else None}
        for f in FACTORS:
            row[f] = boolean(r[f])
            row[f'qwen_{f}'] = boolean(r[f'qwen_{f}'])
        labels[(r['prompt_id'], int(r['seed']))] = row
    return labels


def load_run(root):
    root = Path(root)
    return root, load_prompts(root), read_csv(root / 'pairs.csv'), load_labels(root)


def strict_value(prompt, row):
    values = [row['object'], *[row[f] for f in prompt['active']]]
    return None if None in values else all(values)


def interference(model, setting, prompts, pairs, labels, boot):
    seeds = sorted({k[1] for k in labels})
    rows = []
    for a in FACTORS:
        for b in FACTORS:
            if a == b:
                continue
            items = []
            for pair in pairs:
                if pair['added_factor'] != b or a not in prompts[pair['base_id']]['active']:
                    continue
                for seed in seeds:
                    x = labels.get((pair['base_id'], seed), {})
                    y = labels.get((pair['added_id'], seed), {})
                    if x.get('object') is True and y.get('object') is True and x.get(a) is not None and y.get(a) is not None:
                        items.append((prompts[pair['base_id']]['object'], x[a], y[a]))
            flips = sum(x and not y for _, x, y in items)
            gains = sum(y and not x for _, x, y in items)
            low, high = ci(boot.stat([(o, int(y) - int(x)) for o, x, y in items])[1])
            rows.append({'model': model, 'setting': setting, 'checked': a, 'added': b, 'pairs': len(items),
                         'delta': sum(int(y) - int(x) for _, x, y in items) / len(items),
                         'ci_low': low, 'ci_high': high, 'flips': flips, 'gains': gains,
                         'mcnemar_p': mcnemar_p(flips, gains)})
    for r, p in zip(rows, holm([r['mcnemar_p'] for r in rows])):
        r['mcnemar_p_holm'] = p
    return rows


def compare_models(setting, prompts, base, second):
    rows = []
    for name in ['N', 'P', 'strict']:
        items = []
        for key in base.keys() & second.keys():
            if name == 'strict':
                x, y = strict_value(prompts[key[0]], base[key]), strict_value(prompts[key[0]], second[key])
            elif name in prompts[key[0]]['active']:
                x, y = base[key][name], second[key][name]
            else:
                continue
            if x is not None and y is not None:
                items.append((x, y))
        only_base = sum(x and not y for x, y in items)
        only_second = sum(y and not x for x, y in items)
        rows.append({'setting': setting, 'metric': name, 'shared_jobs': len(items),
                     'base_accuracy': sum(x for x, _ in items) / len(items),
                     'second_accuracy': sum(y for _, y in items) / len(items),
                     'only_base_correct': only_base, 'only_second_correct': only_second,
                     'mcnemar_p': mcnemar_p(only_base, only_second)})
    return rows
